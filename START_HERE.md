# Start Here: EV Battery Health Project

## The project in one sentence

We use measurements collected while a lithium-ion battery is discharging to estimate how much of its original capacity is still available, and then use SHAP and LIME to explain the estimate.

## The real-world problem

An EV battery becomes weaker as it is used. It may still charge to “100%,” but that 100% can represent less energy than when the battery was new.

For example:

- New battery capacity: 2.0 Ah
- Current measured capacity: 1.6 Ah
- State of Health: `1.6 / 2.0 × 100 = 80%`

State of Charge and State of Health are different:

- **State of Charge:** How full is the battery right now?
- **State of Health:** How much capacity remains compared with when it was new?

## What exactly are we predicting?

We predict **State of Health as a percentage**.

The model output can look like this:

```text
Actual battery health:    78.4%
Model estimate:           79.1%
Prediction error:          0.7 percentage points
```

This is a regression problem because the result is a continuous number, not a yes/no category.

## What dataset are we using?

We use the NASA Ames Battery Data Set created for battery degradation and prognostics research.

NASA repeatedly performed three kinds of operations:

1. **Charge:** Put energy into the cell.
2. **Discharge:** Apply a load and take energy out while recording sensors.
3. **Impedance:** Measure internal electrical behavior.

We use four lithium-ion laboratory cells:

| Cell | Role in this project | Discharge cycles |
|---|---|---:|
| B0005 | Training | 168 |
| B0006 | Training | 168 |
| B0007 | Training | 168 |
| B0018 | Final unseen test | 132 |

Total discharge-cycle rows after processing: **636**.

The original files are MATLAB `.mat` files. Each discharge operation contains many sensor readings collected over time. We summarize those readings into one table row per discharge cycle.

## What is one row in our processed table?

One row means:

> “One complete discharge cycle of one physical battery cell.”

Example columns:

| Column | Simple meaning |
|---|---|
| `battery_id` | Which physical cell produced the data |
| `cycle_number` | Which discharge cycle this was |
| `voltage_mean_v` | Average voltage during that discharge |
| `voltage_drop_v` | How far voltage fell during the discharge |
| `current_mean_a` | Average measured current |
| `temperature_max_c` | Highest measured temperature |
| `temperature_rise_c` | How much temperature increased |
| `discharge_duration_s` | How long the discharge lasted |
| `delivered_energy_wh` | Approximate energy supplied during the discharge |
| `soh_pct` | Target: capacity remaining compared with the original 2 Ah |

## How do we prevent cheating?

NASA provides measured capacity. We use it to calculate the correct SOH target.

We do **not** give measured capacity to the model as an input. If we did, the model would effectively receive the answer.

We also keep B0018 completely outside training. This prevents the model from seeing earlier cycles from the same physical battery it is later tested on.

## What does the Random Forest do?

A Random Forest contains many decision trees. Each tree makes an SOH estimate using questions such as:

```text
Was delivered energy low?
Was discharge duration short?
Was this a late cycle?
Was voltage drop large?
```

The forest combines the estimates from all its trees.

It is called a black-box model because hundreds of trees are difficult to understand by reading the model directly.

## Why do we need SHAP?

SHAP explains what pushed a model estimate up or down.

Suppose the model starts from an average estimate of 80% SOH:

```text
Average model estimate                   80%
Low delivered energy                    -7%
Short discharge duration                -2%
High voltage stability                  +1%
Final estimate                          72%
```

The global SHAP chart tells us what matters across many predictions. The waterfall chart explains one battery cycle.

SHAP explains the model’s behavior. It does not prove that a feature physically caused degradation.

## Why also use LIME?

LIME makes many slightly modified examples near one battery cycle, asks the Random Forest to predict them, and builds a small simple model around that case.

It produces rules such as:

```text
Delivered energy <= threshold       pushed health lower
Cycle number > threshold             pushed health lower
Voltage drop <= threshold            pushed health higher
```

LIME is useful for a second local explanation, but it can change depending on its artificial samples. SHAP is the primary explanation method in this project.

## What did the model find?

For the unseen B0018 cell:

- Random Forest MAE: **0.699 SOH percentage points**
- Random Forest RMSE: **0.794**
- Random Forest R²: **0.989**
- Largest health overestimate: **1.682 points**

The strongest Random Forest signals were:

1. Delivered energy
2. Discharge duration
3. Cycle number
4. Voltage drop
5. Current variability

## One honest and important result

Linear regression achieved a lower MAE of **0.350 points**.

That does not make the project unsuccessful. It teaches an important professional lesson:

> A complicated model should not be selected only because it is complicated.

The project explains the Random Forest because the assignment is about black-box explainability. If choosing a production model, we would investigate why the simple model performed better and validate both on more batteries.

## How could an automotive company use this?

After collecting a monitored discharge cycle, the method could support:

- Inspection prioritization
- Fleet-maintenance planning
- Warranty investigation
- Detection of unexpectedly fast degradation
- Comparison of health trends between battery packs

It should not automatically decide to replace a battery.

## Main limitation

The dataset contains four laboratory cells, not complete EV battery packs installed in vehicles. It does not include pack imbalance, road vibration, fast-charging history, cooling-system behavior or real driving loads.

Therefore, describe it as:

> “A proof-of-concept explainable battery-health workflow using public laboratory data.”

Do not describe it as a production-ready EV Battery Management System.
