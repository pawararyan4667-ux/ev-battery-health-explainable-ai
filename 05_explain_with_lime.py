"""
STEP 5 - EXPLAIN ONE PREDICTION WITH LIME

LIME creates many similar artificial examples near one battery cycle and fits a
small, easy-to-read model around that one prediction.

INPUTS:
    The trained Random Forest, training data and B0018 predictions.

OUTPUTS:
    results/lime_local_rules.csv
    results/lime_local_explanation.html
    results/lime_local_explanation.png

MANAGER EXPLANATION:
    "LIME provides a second local explanation. It approximates the Random
    Forest around one selected battery cycle using simple threshold rules."
"""

# joblib loads the model trained in Step 3.
import joblib

# Use a non-interactive plotting backend so the chart can be saved automatically.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pandas reads data and saves the LIME rules.
import pandas as pd

# LimeTabularExplainer handles normal rows-and-columns datasets.
from lime.lime_tabular import LimeTabularExplainer

from config import (
    FEATURES,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_BATTERY,
    TRAIN_BATTERIES,
)


def main():
    """Explain the largest Random Forest health overestimate with LIME."""

    # Load the exact trained model already evaluated and explained with SHAP.
    model = joblib.load(MODEL_DIR / "random_forest.joblib")

    # LIME learns the normal distribution of feature values from training data.
    # B0018 remains separate because it is the unseen test battery.
    data = pd.read_csv(PROCESSED_DATA_DIR / "battery_cycles.csv")
    train = data[data["battery_id"].isin(TRAIN_BATTERIES)]
    test = data[data["battery_id"] == TEST_BATTERY].reset_index(drop=True)
    prediction_results = pd.read_csv(RESULTS_DIR / "b0018_predictions.csv")

    # Select the same meaningful case used for local SHAP: the row where the
    # Random Forest overestimated health by the largest amount.
    row_number = int(prediction_results["error"].idxmax())

    # mode="regression" is required because SOH is a continuous number.
    # discretize_continuous=True converts numeric measurements into readable
    # ranges such as "delivered_energy_wh <= 4.98".
    lime_explainer = LimeTabularExplainer(
        train[FEATURES].values,
        feature_names=FEATURES,
        mode="regression",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )
    # Create 5,000 slightly modified examples near the selected B0018 cycle.
    # The Random Forest predicts them, and LIME fits a small local approximation.
    explanation = lime_explainer.explain_instance(
        test.loc[row_number, FEATURES].values,
        model.predict,
        num_features=10,
        num_samples=5000,
    )

    # Save the rules as CSV for inspection and HTML for an interactive view.
    rules = pd.DataFrame(explanation.as_list(), columns=["local_rule", "contribution"])
    rules.to_csv(RESULTS_DIR / "lime_local_rules.csv", index=False)
    explanation.save_to_file(str(RESULTS_DIR / "lime_local_explanation.html"))

    # Save a static PNG chart suitable for a report or manager update.
    figure = explanation.as_pyplot_figure()
    figure.set_size_inches(9, 5)
    plt.title("LIME explanation for the largest SOH overestimate")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "lime_local_explanation.png", dpi=180)
    plt.close()

    print("\nLIME EXPLANATION")
    print("=" * 45)
    print(rules.to_string(index=False))
    print("\nPositive contribution = pushed estimated health higher.")
    print("Negative contribution = pushed estimated health lower.")
    print("LIME explains only this nearby region, not the whole model.")


# Run LIME only when this file is executed directly.
if __name__ == "__main__":
    main()
