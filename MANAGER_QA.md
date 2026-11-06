# Manager Explanation and Questions

## 60-second project explanation

> I used NASA laboratory data from four lithium-ion battery cells. NASA repeatedly charged and discharged the cells while recording voltage, current, temperature, time and remaining capacity. I converted every discharge cycle into one table row containing summary measurements. The target was State of Health, calculated as current capacity divided by the original 2 Ah capacity. I trained a Random Forest using three cells and tested it on a fourth cell that the model had never seen. The Random Forest’s average error was 0.699 SOH percentage points. I then used SHAP to explain the overall model and individual estimates, and LIME as a second local explanation. The main model signals were delivered energy, discharge duration and cycle number. The main limitation is that these are four laboratory cells, not complete EV battery packs.

## Two-minute explanation

> The engineering problem is battery degradation. A battery can be fully charged but still have less usable capacity than when it was new. That long-term condition is called State of Health. The dataset contains four physical lithium-ion cells named B0005, B0006, B0007 and B0018. During each discharge, NASA recorded many voltage, current and temperature readings. I summarized each discharge into features such as average voltage, voltage drop, maximum temperature, temperature rise, discharge duration and delivered energy.
>
> I calculated the correct State of Health from measured discharge capacity. I did not use measured capacity as an input because that would give the answer to the model. I trained the Random Forest on cells B0005, B0006 and B0007 and reserved B0018 as a completely unseen test cell. This is more realistic than randomly mixing cycles from every cell.
>
> On B0018, the Random Forest had a mean absolute error of 0.699 percentage points and an R-squared value of 0.989. SHAP showed that delivered energy, discharge duration and cycle number were the strongest prediction drivers. A local waterfall chart explained the cycle where the model most overestimated health. LIME was used as a second local check. Linear regression actually performed better, so I would not automatically deploy the Random Forest. The project demonstrates explainability and also the importance of comparing complex models with simpler baselines.

## Likely questions and good answers

### What is the business use case?

The concept could support battery inspection prioritization, fleet-maintenance planning and warranty investigation after a monitored discharge cycle.

### What is the target variable?

State of Health percentage. It measures remaining capacity compared with the original capacity.

### Why is this regression?

The model predicts a continuous number such as 78.6%, not a category.

### What is one row in the processed dataset?

One complete discharge cycle from one physical cell.

### Why did you use a Random Forest?

It can learn nonlinear relationships and feature interactions without requiring a fixed equation. It is also a suitable black-box model for demonstrating SHAP and LIME.

### How did you split the data?

I trained on B0005, B0006 and B0007 and tested only on B0018. No B0018 cycle was used for training.

### Why is that split better than a random split?

With a random split, early cycles from B0018 could appear in training and later cycles from the same cell in testing. The model could partially memorize that cell’s degradation pattern.

### What does MAE 0.699 mean?

Across B0018, the Random Forest estimate differed from measured SOH by approximately 0.699 percentage points on average.

### What does R² 0.989 mean?

The model represented about 98.9% of the variation in B0018 SOH. It does not mean that 98.9% of individual predictions were correct.

### What did SHAP tell you?

Delivered energy was the strongest model signal. Discharge duration and cycle number were next. Lower energy and shorter duration generally pushed estimated health downward.

### What did LIME tell you?

LIME explained one selected cycle using simple local ranges. It provided a second view of which measurements pushed that one estimate higher or lower.

### Are SHAP results causal?

No. SHAP explains what the model used. It does not prove that changing one measurement would physically cause a specific SOH change.

### Why did linear regression perform better?

The relationship between full-cycle energy, duration and measured capacity is strong and comparatively simple in this controlled experiment. The Random Forest’s piecewise predictions may generalize less smoothly to the held-out cell. More cell-level validation would be required before selecting either model.

### Does that make the Random Forest project invalid?

No. The assignment requires explaining a black-box model, which the Random Forest satisfies. The comparison demonstrates responsible model selection rather than hiding an inconvenient result.

### What is target leakage here?

Measured capacity is used to calculate SOH. If capacity were also used as an input feature, the model would be given the answer.

### Is this ready for an EV production system?

No. It is a proof of concept using four laboratory cells. A production system needs pack-level fleet data, more battery chemistries, thermal gradients, cell imbalance, driving conditions, uncertainty estimates and prospective validation.

### What would you do next?

Collect data from many vehicle packs, validate by holding out entire packs, add charging and impedance features, quantify uncertainty and evaluate errors specifically near service thresholds.
