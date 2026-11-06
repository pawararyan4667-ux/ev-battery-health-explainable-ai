"""
STEP 3 - TRAIN AND TEST THE RANDOM FOREST

Training batteries: B0005, B0006, B0007
Unseen test battery: B0018

INPUT:
    data/processed/battery_cycles.csv

OUTPUTS:
    models/random_forest.joblib
    results/b0018_predictions.csv
    results/model_metrics.json
    results/actual_vs_predicted.png

MANAGER EXPLANATION:
    "The model learned from three physical cells and was evaluated on a fourth
    cell that was never used during training. This is more honest than randomly
    mixing cycles from every cell."
"""

# json saves model results in a standard machine-readable format.
import json

# joblib saves the trained Random Forest so SHAP and LIME can load it later.
import joblib

# Agg is a non-interactive plotting backend. It lets the script save PNG files
# even when no chart window is available.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pandas reads the processed cycle-level table.
import pandas as pd

# RandomForestRegressor is the black-box model required for the project.
from sklearn.ensemble import RandomForestRegressor

# LinearRegression is a simple baseline used for comparison.
from sklearn.linear_model import LinearRegression

# These functions measure prediction error and overall model fit.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    FEATURES,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_BATTERY,
    TRAIN_BATTERIES,
    create_project_folders,
)


def calculate_metrics(actual, predicted):
    """Calculate three commonly used regression metrics."""

    return {
        # MAE: average absolute distance between actual and predicted SOH.
        "MAE": mean_absolute_error(actual, predicted),

        # RMSE: similar to MAE, but large errors receive more penalty.
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,

        # R²: proportion of target variation represented by the model.
        "R2": r2_score(actual, predicted),
    }


def main():
    """Load the data, train two models and evaluate the unseen battery."""

    create_project_folders()

    # Load the table created by 02_prepare_data.py.
    data_path = PROCESSED_DATA_DIR / "battery_cycles.csv"
    data = pd.read_csv(data_path)

    # Cell-held-out split:
    # - the Random Forest can learn from B0005, B0006 and B0007;
    # - it cannot see any B0018 row until final testing.
    train = data[data["battery_id"].isin(TRAIN_BATTERIES)].copy()
    test = data[data["battery_id"] == TEST_BATTERY].copy()

    # X contains the input features and y contains the target SOH.
    # battery_id is not used as a feature, preventing the model from simply
    # memorizing a battery name.
    x_train, y_train = train[FEATURES], train["soh_pct"]
    x_test, y_test = test[FEATURES], test["soh_pct"]

    print("\nWHY THIS SPLIT IS IMPORTANT")
    print("=" * 45)
    print("The model learns from three physical cells:", TRAIN_BATTERIES)
    print("It is tested on a different cell:", TEST_BATTERY)
    print("This checks whether it can work on a battery it has never seen.\n")

    # Create the main black-box model.
    # n_estimators=500 means the forest combines 500 decision trees.
    # max_depth and min_samples_leaf reduce excessive memorization.
    # random_state makes repeated runs reproducible.
    random_forest = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    # Learn relationships between the training measurements and SOH.
    random_forest.fit(x_train, y_train)

    # Estimate the health of every discharge cycle from unseen cell B0018.
    rf_predictions = random_forest.predict(x_test)

    # A simple model is included so we do not assume complexity is always better.
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    linear_predictions = linear_model.predict(x_test)

    # Calculate the same metrics for a fair model comparison.
    rf_metrics = calculate_metrics(y_test, rf_predictions)
    linear_metrics = calculate_metrics(y_test, linear_predictions)

    print("RANDOM FOREST RESULTS")
    print(f"Average error (MAE) : {rf_metrics['MAE']:.3f} SOH points")
    print(f"RMSE                : {rf_metrics['RMSE']:.3f}")
    print(f"R-squared           : {rf_metrics['R2']:.3f}")
    print("\nLINEAR REGRESSION COMPARISON")
    print(f"Average error (MAE) : {linear_metrics['MAE']:.3f} SOH points")
    print("\nExample meaning: an MAE of 0.699 means the Random Forest's")
    print("health estimate is about 0.699 percentage points away on average.")

    # Store each B0018 prediction next to its measured value. A positive error
    # means the model overestimated health; a negative error means it
    # underestimated health.
    results = test[["battery_id", "cycle_number", "soh_pct"]].copy()
    results["predicted_soh_pct"] = rf_predictions
    results["error"] = results["predicted_soh_pct"] - results["soh_pct"]
    results["absolute_error"] = results["error"].abs()
    # Save the prediction table and trained model for later inspection.
    results.to_csv(RESULTS_DIR / "b0018_predictions.csv", index=False)

    joblib.dump(random_forest, MODEL_DIR / "random_forest.joblib")

    # Save the key experiment settings and metrics so the results are recorded
    # outside the terminal output.
    summary = {
        "training_batteries": TRAIN_BATTERIES,
        "test_battery": TEST_BATTERY,
        "training_rows": len(train),
        "test_rows": len(test),
        "random_forest": rf_metrics,
        "linear_regression": linear_metrics,
    }
    (RESULTS_DIR / "model_metrics.json").write_text(json.dumps(summary, indent=2))

    # Plot actual and predicted SOH across the entire life of B0018.
    # Closely overlapping lines indicate close predictions.
    plt.figure(figsize=(10, 5))
    plt.plot(test["cycle_number"], y_test, label="Actual SOH", linewidth=2.5)
    plt.plot(test["cycle_number"], rf_predictions, label="Predicted SOH", linewidth=2)
    plt.xlabel("Discharge cycle number")
    plt.ylabel("State of Health (%)")
    plt.title("Actual and predicted battery health for unseen cell B0018")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close()

    # Locate the largest positive error. This is especially important because
    # overestimating health can make a degraded battery appear healthier.
    worst = results.loc[results["error"].idxmax()]
    print("\nMOST IMPORTANT ERROR TO DISCUSS")
    print(f"Cycle                 : {int(worst['cycle_number'])}")
    print(f"Actual health          : {worst['soh_pct']:.3f}%")
    print(f"Predicted health       : {worst['predicted_soh_pct']:.3f}%")
    print(f"Model overestimated by : {worst['error']:.3f} points")


# Run the training workflow only when this file is executed directly.
if __name__ == "__main__":
    main()
