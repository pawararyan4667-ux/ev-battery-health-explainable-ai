"""
STEP 4 - EXPLAIN THE RANDOM FOREST WITH SHAP

Global explanation: what matters across all B0018 predictions?
Local explanation: why did the model produce one particular estimate?

INPUTS:
    The trained Random Forest, processed battery data and B0018 predictions.

OUTPUTS:
    results/shap_global_summary.png
    results/shap_feature_importance.csv
    results/shap_local_waterfall.png

MANAGER EXPLANATION:
    "SHAP shows which measurements moved the model's health estimate higher or
    lower. It explains what the model used; it does not prove physical cause."
"""

# joblib loads the Random Forest saved by 03_train_model.py.
import joblib

# Use a non-interactive backend because charts are saved as PNG files.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NumPy calculates average absolute SHAP values.
import numpy as np

# pandas reads the processed data and saves the importance table.
import pandas as pd

# SHAP calculates feature contributions for the tree model.
import shap

from config import FEATURES, MODEL_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, TEST_BATTERY


def main():
    """Create global and local explanations for unseen battery B0018."""

    # Load the already-trained model. We explain exactly the same model that was
    # evaluated in Step 3; it is not retrained here.
    model = joblib.load(MODEL_DIR / "random_forest.joblib")

    # Select only B0018, the completely unseen test cell.
    data = pd.read_csv(PROCESSED_DATA_DIR / "battery_cycles.csv")
    test = data[data["battery_id"] == TEST_BATTERY].copy()
    x_test = test[FEATURES]

    # TreeExplainer is designed for tree-based models such as Random Forest.
    # The returned object contains one SHAP contribution per feature per row.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x_test)

    # Global plot: every dot is one B0018 cycle.
    # Horizontal position shows whether the feature pushed estimated SOH up or
    # down. Color represents whether the original feature value was high or low.
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("Which measurements influenced battery-health estimates?")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_global_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Mean absolute SHAP value measures average influence regardless of whether
    # the feature pushed a prediction upward or downward.
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "average_shap_importance": np.abs(shap_values.values).mean(axis=0),
        }
    ).sort_values("average_shap_importance", ascending=False)
    importance.to_csv(RESULTS_DIR / "shap_feature_importance.csv", index=False)

    # Use the prediction table to select the largest health overestimate rather
    # than choosing a convenient example by hand.
    predictions = pd.read_csv(RESULTS_DIR / "b0018_predictions.csv")
    worst_row = predictions["error"].idxmax()

    # The waterfall begins at the model's average prediction and then shows how
    # the selected cycle's measurements move the estimate to its final value.
    shap.plots.waterfall(shap_values[worst_row], max_display=10, show=False)
    cycle = int(test.iloc[worst_row]["cycle_number"])
    plt.title(f"Why the model predicted B0018 cycle {cycle}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_local_waterfall.png", dpi=180, bbox_inches="tight")
    plt.close()

    print("\nSHAP EXPLANATION")
    print("=" * 45)
    print("Top five measurements used by the model:")
    for number, row in enumerate(importance.head(5).itertuples(), start=1):
        print(f"{number}. {row.feature}: {row.average_shap_importance:.3f}")

    print("\nHow to explain SHAP to a manager:")
    print("SHAP starts from the model's average health estimate.")
    print("It then shows which measurements pushed one estimate higher")
    print("and which measurements pushed it lower.")
    print("It explains the model; it does not prove physical causation.")


# Run SHAP only when this file is executed directly.
if __name__ == "__main__":
    main()
