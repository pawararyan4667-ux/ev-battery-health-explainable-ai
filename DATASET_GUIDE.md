# Dataset Guide in Simple Words

## Dataset identity

**Name:** NASA Ames Prognostics Center of Excellence Battery Data Set  
**Original authors:** B. Saha and K. Goebel  
**Original year:** 2007  
**Official catalog:** https://data.nasa.gov/dataset/li-ion-battery-aging-datasets  
**Files used:** B0005.mat, B0006.mat, B0007.mat and B0018.mat

The files in this project are NASA-origin files retrieved from a public mirror because the older NASA archive was unavailable when the project was assembled.

## File checksums

Checksums allow another person to verify that their copy is exactly the same as ours.

```text
B0005.mat  0eae4585baf3f200c09fe24c5ab884f1889679fc75206ca1aa19da704104f0b0
B0006.mat  fa818ab4db5db8ab21e910b6dd6c3e20d3761bb9672089e1a4de8f96074616c5
B0007.mat  d022afa086efaf54ab5b63f05220f5be178c8027e2fa8d68589db0e85a441a3b
B0018.mat  d1e6c923a43ea1c9666b3a90bbb521757a067fd60b4d17dbfaa49c50b179da69
```

## What NASA physically did

NASA used commercial lithium-ion cells on a laboratory test system. The cells were repeatedly charged and discharged. During the experiments, instruments recorded electrical and temperature measurements.

Repeated cycling caused battery capacity to decline. This gives us examples covering healthy and degraded states.

## Raw file structure

Each battery file contains a list of operations. Every operation includes:

- Operation type: charge, discharge or impedance
- Ambient temperature
- Start time
- A nested sensor-data section

For a discharge operation, important fields include:

- `Voltage_measured`
- `Current_measured`
- `Temperature_measured`
- `Current_load`
- `Voltage_load`
- `Time`
- `Capacity`

Voltage, current and temperature are arrays because they were measured repeatedly throughout one discharge.

## How the raw data becomes a machine-learning table

Raw discharge data:

```text
Cycle 10 voltage = [4.18, 4.16, 4.15, ..., 2.70]
Cycle 10 current = [0.00, -1.95, -1.98, ..., -1.90]
Cycle 10 temperature = [24.0, 24.1, ..., 36.8]
```

Processed row:

```text
battery_id = B0005
cycle_number = 10
voltage_mean_v = 3.52
voltage_drop_v = 1.48
temperature_max_c = 36.8
discharge_duration_s = 3500
delivered_energy_wh = 6.2
soh_pct = 94.5
```

## Why only discharge cycles are used

Measured discharge capacity directly supports calculating State of Health. Charge and impedance operations are valuable, but joining them safely to the correct discharge cycle would add complexity that is unnecessary for this first explainability project.

## Target calculation

NASA’s cells have an original rated capacity of 2 Ah in this project.

```text
SOH (%) = measured capacity / 2 Ah × 100
```

If measured capacity is 1.4 Ah:

```text
SOH = 1.4 / 2.0 × 100 = 70%
```

## Data split

```text
B0005 + B0006 + B0007  → training data
B0018                  → final test data
```

This is called a cell-held-out test. It is harder and more realistic than randomly placing cycles from every cell in both training and testing.

## Questions a reviewer may raise

### Is this real EV road data?

No. It is real laboratory lithium-ion cell data with EV relevance. It is not a complete EV pack or road fleet dataset.

### Why are only four cells used?

These four cells are a commonly used and manageable subset of the NASA battery-aging data. The small number of physical cells is also the project’s biggest limitation.

### Why not give capacity to the model?

Capacity is used to calculate the answer, SOH. Giving it to the model would cause target leakage.

### Can the estimate be produced at every instant?

Not with the current feature set. Discharge duration and delivered energy are known after the monitored discharge finishes. This is a post-cycle health-estimation project.
