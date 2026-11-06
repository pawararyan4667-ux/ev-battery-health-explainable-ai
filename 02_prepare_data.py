"""
STEP 2 - CONVERT NASA TIME-SERIES FILES INTO A SIMPLE TABLE

One output row = one complete discharge cycle.
The output is saved as data/processed/battery_cycles.csv.

WHY THIS STEP IS NEEDED:
    A machine-learning model expects a table, but each NASA discharge contains
    long arrays of voltage, current and temperature measurements. This script
    summarizes those arrays into useful cycle-level features.

INPUT:
    Four raw NASA .mat files.

OUTPUT:
    data/processed/battery_cycles.csv with 636 rows and 17 columns.

MANAGER EXPLANATION:
    "I converted every complete discharge into one table row. Each row contains
    summaries such as average voltage, maximum temperature, discharge duration
    and delivered energy."
"""

# NumPy performs numerical calculations on the sensor arrays.
import numpy as np

# pandas builds and saves the final table.
import pandas as pd

# loadmat opens the original MATLAB files.
from scipy.io import loadmat

from config import (
    BATTERY_IDS,
    ORIGINAL_CAPACITY_AH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    create_project_folders,
)


def as_numbers(values):
    """Convert a MATLAB sensor array into a clean one-dimensional array."""

    # reshape(-1) flattens the data into one dimension.
    array = np.asarray(values, dtype=float).reshape(-1)

    # Remove invalid values such as NaN or infinity before calculating means,
    # minimums and maximums.
    return array[np.isfinite(array)]


def calculate_energy_wh(voltage, current, time_seconds):
    """Approximate delivered energy by integrating voltage × current over time."""

    # All three arrays should normally have the same length. Using the shortest
    # length makes the calculation safe if a file contains a small mismatch.
    length = min(len(voltage), len(current), len(time_seconds))
    voltage = voltage[:length]

    # Discharge current can be recorded as negative. Absolute current represents
    # its magnitude for the delivered-energy calculation.
    current = np.abs(current[:length])
    time_seconds = time_seconds[:length]

    # Voltage × current gives power. Integrating power over seconds and dividing
    # by 3600 converts the result into watt-hours.
    return np.trapezoid(voltage * current, time_seconds) / 3600


def extract_battery(battery_id):
    """Create one table row for every discharge cycle of one battery."""

    # Open one battery file and retrieve its sequence of operations.
    path = RAW_DATA_DIR / f"{battery_id}.mat"
    operations = loadmat(path, simplify_cells=True)[battery_id]["cycle"]

    # rows will hold dictionaries that later become a pandas DataFrame.
    rows = []
    discharge_number = 0

    for operation in operations:
        # This first version of the project uses only discharge operations.
        # Charge and impedance data remain available for future extensions.
        if str(operation["type"]).lower() != "discharge":
            continue

        # Count only discharge cycles, ignoring charge and impedance operations.
        discharge_number += 1
        data = operation["data"]

        # Convert raw MATLAB arrays into clean one-dimensional numeric arrays.
        voltage = as_numbers(data["Voltage_measured"])
        current = as_numbers(data["Current_measured"])
        temperature = as_numbers(data["Temperature_measured"])
        load_voltage = as_numbers(data["Voltage_load"])
        time_seconds = as_numbers(data["Time"])

        # Capacity is the measured amount of charge the cell supplied during
        # this discharge. It is used only to construct the correct target.
        measured_capacity_ah = float(np.asarray(data["Capacity"]).squeeze())

        # SOH says how much capacity remains compared with the original 2 Ah.
        # Example: 1.6 Ah / 2.0 Ah × 100 = 80% SOH.
        soh_pct = measured_capacity_ah / ORIGINAL_CAPACITY_AH * 100

        # Build one machine-learning row for the current discharge cycle.
        # The model receives the engineered measurements below, but it does not
        # receive measured_capacity_ah because that would reveal the answer.
        rows.append(
            {
                "battery_id": battery_id,
                "cycle_number": discharge_number,
                "ambient_temperature_c": float(operation["ambient_temperature"]),
                # Average, minimum, variation and total drop describe voltage.
                "voltage_mean_v": voltage.mean(),
                "voltage_min_v": voltage.min(),
                "voltage_std_v": voltage.std(),
                "voltage_drop_v": voltage[0] - voltage[-1],
                # Current mean and variation describe the applied electrical load.
                "current_mean_a": current.mean(),
                "current_std_a": current.std(),
                # Thermal summaries describe how hot the cell became.
                "temperature_mean_c": temperature.mean(),
                "temperature_max_c": temperature.max(),
                "temperature_rise_c": temperature.max() - temperature[0],
                "load_voltage_mean_v": load_voltage.mean(),
                "load_voltage_min_v": load_voltage.min(),
                # Duration says how long the cell supported the discharge load.
                "discharge_duration_s": time_seconds.max() - time_seconds.min(),

                # Delivered energy combines voltage, current and time.
                "delivered_energy_wh": calculate_energy_wh(voltage, current, time_seconds),

                # soh_pct is the value the model will learn to predict.
                "soh_pct": soh_pct,
            }
        )

    # Convert all cycle dictionaries for this battery into a table.
    return pd.DataFrame(rows)


def main():
    """Process all four cells and save one combined CSV file."""

    # Ensure the data, models and results folders exist.
    create_project_folders()

    # Extract each cell independently and then place the tables underneath one
    # another. The result contains all 636 discharge cycles.
    tables = [extract_battery(battery_id) for battery_id in BATTERY_IDS]
    battery_cycles = pd.concat(tables, ignore_index=True)

    # Save the processed table for the model-training script.
    output_path = PROCESSED_DATA_DIR / "battery_cycles.csv"
    battery_cycles.to_csv(output_path, index=False)

    print("\nPROCESSED DATA CREATED")
    print("=" * 45)
    print("Output:", output_path)
    print("Rows:", len(battery_cycles))
    print("Columns:", len(battery_cycles.columns))
    print("\nRows per battery:")
    print(battery_cycles.groupby("battery_id").size())
    print("\nSOH range per battery:")
    print(battery_cycles.groupby("battery_id")["soh_pct"].agg(["max", "min"]).round(2))
    print("\nImportant: measured capacity was used to calculate the target SOH,")
    print("but measured capacity is NOT given to the model as an input.")


# Run main() only when this script is executed directly.
if __name__ == "__main__":
    main()
