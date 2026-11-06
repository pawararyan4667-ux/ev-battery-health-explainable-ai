"""
MAIN FILE FOR THE EV BATTERY HEALTH PROJECT

Run this file when you want to execute the complete project:

    python run_all.py

This file does not contain the machine-learning calculations itself. Instead,
it acts like a project controller: it launches the five numbered Python files
in the correct order. Keeping the stages separate makes the project easier to
learn, test and explain to another person.
"""

# subprocess lets one Python file start another Python file and wait for it.
import subprocess

# sys gives us the path of the Python interpreter that is currently running.
# Using sys.executable ensures every stage uses the same VS Code virtual
# environment and the same installed packages.
import sys


# These scripts form the complete data-science workflow.
# Their order matters because each stage produces something required by the
# next stage. For example, the model cannot be trained until the processed CSV
# has been created by 02_prepare_data.py.
SCRIPTS = [
    # Step 1: Open the original NASA .mat files and print what they contain.
    "01_understand_dataset.py",

    # Step 2: Convert raw voltage, current and temperature arrays into one
    # machine-learning row for every complete battery discharge cycle.
    "02_prepare_data.py",

    # Step 3: Train the Random Forest on B0005, B0006 and B0007, then evaluate
    # it on the completely unseen B0018 battery.
    "03_train_model.py",

    # Step 4: Use SHAP to explain the model globally and explain one individual
    # battery-health prediction with a waterfall chart.
    "04_explain_with_shap.py",

    # Step 5: Use LIME as a second local explanation for the most important
    # prediction error.
    "05_explain_with_lime.py",
]


def main():
    """Run every project stage in sequence."""

    # Loop through the filenames listed above, starting with Step 1.
    for script in SCRIPTS:
        # Print a clear heading so the terminal shows which stage is running.
        print("\n" + "#" * 70)
        print("RUNNING", script)
        print("#" * 70)

        # Start the selected script with the current Python interpreter.
        # check=True means the complete workflow stops immediately if a stage
        # fails. This prevents later stages from using missing or incorrect
        # files and makes errors easier to identify.
        subprocess.run([sys.executable, script], check=True)

    # This message appears only when all five stages finish successfully.
    print("\nProject completed. Open the results folder to see all charts.")


# Python sets __name__ to "__main__" when this file is run directly.
# This prevents main() from starting automatically if run_all.py is imported
# into another Python program.
if __name__ == "__main__":
    main()
