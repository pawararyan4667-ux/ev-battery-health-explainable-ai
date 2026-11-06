"""
STEP 1 - UNDERSTAND THE NASA BATTERY FILES

Run in the VS Code terminal:
    python 01_understand_dataset.py

This script does not train a model. It only shows what is inside the dataset.

INPUT:
    Four NASA MATLAB files: B0005.mat, B0006.mat, B0007.mat and B0018.mat

OUTPUT:
    A readable description printed in the terminal. No file is created.

 EXPLANATION:
    "I first inspected the original data before cleaning it or training a model.
    Each file represents one physical battery cell and contains charge,
    discharge and impedance operations."
"""

# scipy.io.loadmat reads MATLAB .mat files and converts their contents into
# normal Python dictionaries, lists and NumPy arrays.
from scipy.io import loadmat

# BATTERY_IDS contains the four cells used in the project.
# RAW_DATA_DIR tells Python where the original NASA files are stored.
from config import BATTERY_IDS, RAW_DATA_DIR


def main():
    """Inspect every raw battery file and print a simple summary."""

    print("\nNASA BATTERY DATASET - SIMPLE INSPECTION")
    print("=" * 48)
    print("Each .mat file represents one physical lithium-ion cell.")
    print("Inside each file are charge, discharge and impedance operations.\n")

    # Repeat the following inspection for all four physical battery cells.
    for battery_id in BATTERY_IDS:
        # Construct a path such as data/raw/B0005.mat.
        path = RAW_DATA_DIR / f"{battery_id}.mat"

        # Stop with an understandable message if a raw file is missing.
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. See HOW_TO_RUN_IN_VSCODE.md")

        # The outer .mat dictionary contains a key that matches the battery ID.
        # The "cycle" section is a sequence of experimental operations.
        battery = loadmat(path, simplify_cells=True)[battery_id]
        cycles = battery["cycle"]

        # Start counters at zero, then count each experimental operation type.
        counts = {"charge": 0, "discharge": 0, "impedance": 0}
        for cycle in cycles:
            cycle_type = str(cycle["type"]).lower()
            counts[cycle_type] = counts.get(cycle_type, 0) + 1

        # Display a short summary for the current battery.
        print(f"{battery_id}")
        print(f"  Total recorded operations : {len(cycles)}")
        print(f"  Charge operations         : {counts.get('charge', 0)}")
        print(f"  Discharge operations      : {counts.get('discharge', 0)}")
        print(f"  Impedance operations      : {counts.get('impedance', 0)}")

    # Look more closely at the first discharge operation from B0005 so the
    # user can see which sensor measurements exist in the raw dataset. 
    example_path = RAW_DATA_DIR / "B0005.mat"
    example_cycles = loadmat(example_path, simplify_cells=True)["B0005"]["cycle"]

    # next(...) returns the first operation whose type is "discharge".
    example = next(c for c in example_cycles if str(c["type"]).lower() == "discharge")

    print("\nEXAMPLE DISCHARGE RECORD FROM B0005")
    print("=" * 48)
    print("Ambient temperature:", example["ambient_temperature"], "degrees C")
    print("Available sensor fields:")
    # The keys printed here include voltage, current, temperature, time and
    # capacity. Most of these values are arrays measured throughout the cycle.
    for field in example["data"].keys():
        print("  -", field)

    print("\nWhat one discharge operation means:")
    print("NASA applied a load to the battery and recorded how voltage,")
    print("current and temperature changed until the discharge ended.")


# This condition calls main() only when this file is run directly.
# It prevents the inspection from starting automatically if another Python
# file imports something from this script.
if __name__ == "__main__":
    main()
