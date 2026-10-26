# Survival Analysis for Predicting Customer Response Timing

## 1. Project Overview

This project applies survival analysis to the Complete Journey retail and marketing dataset released by 84.51° to investigate household coupon-redemption patterns and predict response timing. Kaplan–Meier analysis, the Cox Proportional Hazards model, and Random Survival Forest (RSF) are used to explore redemption behaviour and evaluate predictive performance while accounting for censored observations.

## 2. Objectives

- Investigate coupon-redemption patterns across marketing campaigns.
- Construct a household–campaign analytical dataset using historical purchasing behaviour.
- Estimate redemption timing using survival analysis.
- Compare Cox PH and RSF using the concordance index (C-index) and time-dependent AUC.
- Discuss the limitations and potential marketing applications of the results.

## 3. Dataset and Cohort Construction

The Complete Journey dataset contains retail transactions, campaign assignments and descriptions, coupon redemptions, household demographics, and product information.

The 2017 transaction data contained 1,469,307 product-level records spanning 1 January to 31 December 2017. Transaction line items were aggregated into household shopping baskets to represent shopping trips accurately. After selecting 19 campaigns fully contained within the observation period, the final cohort comprised **5,657 household–campaign observations from 1,531 unique households**.

Historical features included prior shopping frequency, total expenditure, average basket value, and days since the last purchase. Only transactions preceding campaign start were used to construct these predictors.

**Data source:** [84.51°](https://www.8451.com/) | [Complete Journey repository and documentation](https://github.com/bradleyboehmke/completejourney)

The dataset is not included in this repository. Users must obtain the data separately and configure the expected input files and paths before running the scripts.

## 4. Methodology

Coupon redemption was defined as the event. The time origin was campaign start, and observations without a recorded redemption by the campaign end were treated as censored.

Two candidate outcomes were compared: coupon redemption and any purchase. Coupon redemption was selected as the primary outcome because it directly represents the response of interest.

The analysis included:

1. **Exploratory Data Analysis (EDA):** Examined event rates, duration distributions, campaign types, and missingness.
2. **Kaplan–Meier analysis:** Estimated redemption-free survival over time and compared campaign types.
3. **Cox Proportional Hazards:** Modelled redemption hazard and provided interpretable predictor associations.
4. **Random Survival Forest:** Captured potentially nonlinear relationships and interactions.
5. **Model evaluation:** Used a household-grouped train/test split, C-index, and time-dependent AUC at 7, 14, and 30 days.

## 5. Key Findings

- The cohort contained 746 observed redemptions, representing **13.19%** of observations.
- Type A had the highest observed redemption rate (16.56%), compared with Type B (7.08%) and Type C (6.98%).
- Cox PH achieved a test C-index of **0.7211**, while RSF achieved **0.7205**.
- RSF produced higher time-dependent AUC at 7 days (0.7193) and 14 days (0.7692). At 30 days, the scores were closer: 0.7449 for RSF and 0.7393 for Cox PH.

The models showed similar overall ranking performance, with no clear overall winner. These findings describe predictive patterns rather than causal campaign effects.

## 6. Project Structure

```text
CompleteJourneySurvival/
├── README.md
├── data/
│   ├── analytical_dataset.csv
│   └── model_dataset.csv
├── eda_results/
├── model_results/
└── Python analysis scripts
```

This is a simplified overview. Refer to the actual project directory for the complete list of scripts and generated outputs.

## 7. Requirements and Execution

**Requirements:** Python 3.12, pandas, NumPy, Matplotlib, scikit-learn, lifelines, and scikit-survival (`sksurv`). Use VS Code or Jupyter Notebook to inspect and run the analysis.

On Windows, activate the existing project environment from the project directory:

```bash
.\.venv312\Scripts\activate
```

Install any missing dependencies in that environment using `python -m pip install package-name`, replacing `package-name` with the required package. Ensure that VS Code uses the same Python interpreter.

Run the relevant Python scripts from the project directory, following the data paths specified in each script. Exploratory outputs are stored in `eda_results/`, while model outputs and evaluation results are stored in `model_results/`.

## 8. Limitations and Conclusion

The historical look-back period varied across campaigns, demographic information contained substantial missingness, and some households contributed observations to multiple campaigns. Eight zero-day redemption events were excluded during model fitting, and evaluation relied on a single train/test split. The 30-day and other horizon-specific metrics also require interpretation within the available follow-up and censoring support.

Overall, the project demonstrates the application of survival analysis to coupon-redemption timing. Further validation and assessment of incremental redemptions, campaign costs, and practical decision benefits would be necessary before real-world deployment.

**Author:** Tahreem Sadia Abbasi
