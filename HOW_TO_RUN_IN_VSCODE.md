# How to Run This Project in Visual Studio Code

## 1. Open the project

In VS Code, select:

```text
File → Open Folder → ev_battery_vscode_project
```

## 2. Install Python

Install Python 3.11 or 3.12 if it is not already installed. In VS Code, install Microsoft’s Python extension.

## 3. Create a virtual environment

Open the VS Code terminal and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

## 4. Install the packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run one step at a time

This is the recommended learning route:

```powershell
python 01_understand_dataset.py
python 02_prepare_data.py
python 03_train_model.py
python 04_explain_with_shap.py
python 05_explain_with_lime.py
```

Do not move to the next script until you understand the output of the current one.

## 6. Run everything together later

```powershell
python run_all.py
```

You can also press `F5` and choose **Run complete battery project**.

## 7. Where results appear

Generated files are saved under:

```text
results/
```

Important files:

- `actual_vs_predicted.png`
- `shap_global_summary.png`
- `shap_local_waterfall.png`
- `lime_local_explanation.png`
- `model_metrics.json`
- `b0018_predictions.csv`

## 8. Correct script order

```text
Raw NASA files
      ↓
01 Understand the files
      ↓
02 Convert time-series readings into a table
      ↓
03 Train and test the model
      ↓
04 Explain with SHAP
      ↓
05 Cross-check one case with LIME
```

## Common errors

### `ModuleNotFoundError`

Activate `.venv` and run:

```powershell
pip install -r requirements.txt
```

### Raw `.mat` file missing

Confirm these exist under `data/raw/`:

```text
B0005.mat
B0006.mat
B0007.mat
B0018.mat
```

### SHAP or LIME chart does not open

The scripts save charts as PNG files instead of requiring an interactive window. Open them from the `results` folder.
