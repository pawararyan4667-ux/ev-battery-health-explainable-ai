"""One place for project paths and settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

BATTERY_IDS = ["B0005", "B0006", "B0007", "B0018"]
TRAIN_BATTERIES = ["B0005", "B0006", "B0007"]
TEST_BATTERY = "B0018"
ORIGINAL_CAPACITY_AH = 2.0
RANDOM_STATE = 42

FEATURES = [
    "cycle_number",
    "ambient_temperature_c",
    "voltage_mean_v",
    "voltage_min_v",
    "voltage_std_v",
    "voltage_drop_v",
    "current_mean_a",
    "current_std_a",
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_rise_c",
    "load_voltage_mean_v",
    "load_voltage_min_v",
    "discharge_duration_s",
    "delivered_energy_wh",
]


def create_project_folders():
    """Create folders that hold generated files."""
    for folder in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
