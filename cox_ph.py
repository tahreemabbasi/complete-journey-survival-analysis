
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter


# ============================================================
# 1. FILE PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_FILE = PROJECT_DIR / "data" / "model_dataset.csv"

OUTPUT_DIR = PROJECT_DIR / "model_results" / "cox_ph"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

required_columns = ["duration_days", "event"]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["duration_days"] = pd.to_numeric(
    df["duration_days"], errors="coerce"
)
df["event"] = pd.to_numeric(df["event"], errors="coerce")

df = df.dropna(subset=["duration_days", "event"]).copy()
df = df[df["event"].isin([0, 1])].copy()
df = df[df["duration_days"] >= 0].copy()
df["event"] = df["event"].astype(int)

# Cox regression requires strictly positive durations in this analysis.
# Keep zero-day observations in the audit, but exclude them from fitting.
zero_day_count = int((df["duration_days"] == 0).sum())
zero_day_events = int(
    ((df["duration_days"] == 0) & (df["event"] == 1)).sum()
)

df = df[df["duration_days"] > 0].copy()

if df.empty or df["event"].sum() == 0:
    raise ValueError(
        "No valid positive-duration observations with redemption events."
    )

with open(
    OUTPUT_DIR / "cohort_audit.txt", "w", encoding="utf-8"
) as file:
    file.write("Cox PH cohort audit\n")
    file.write("===================\n")
    file.write(f"Rows used for fitting: {len(df)}\n")
    file.write(f"Redemption events used: {int(df['event'].sum())}\n")
    file.write(
        f"Censored observations used: {int((df['event'] == 0).sum())}\n"
    )
    file.write(f"Zero-day rows excluded: {zero_day_count}\n")
    file.write(f"Zero-day redemption events excluded: {zero_day_events}\n")

print("Rows used:", len(df))
print("Redemption events:", int(df["event"].sum()))
print("Censored observations:", int((df["event"] == 0).sum()))
print("Zero-day rows excluded:", zero_day_count)


# ============================================================
# 3. SELECT PREDICTORS
# ============================================================

# Numeric variables: values where a numeric difference is meaningful.
numeric_candidates = [
    "prior_trip_count",
    "prior_total_spend",
    "prior_avg_basket_value",
    "days_since_last_purchase",
    "kids_count",
    "demographics_missing",
]

# Categorical variables: categories or grouped values.
categorical_candidates = [
    "campaign_type",
    "age",
    "income",
    "home_ownership",
    "marital_status",
    "household_comp",
    "household_size",
    "has_prior_purchase",
]

numeric_features = [
    col for col in numeric_candidates if col in df.columns
]

categorical_features = [
    col for col in categorical_candidates if col in df.columns
]

if not numeric_features and not categorical_features:
    raise ValueError(
        "None of the expected predictor columns were found. "
        "Check the column names in model_dataset.csv."
    )

# Keep only the outcome, optional household cluster ID, and predictors.
keep_columns = (
    ["duration_days", "event"]
    + (["household_id"] if "household_id" in df.columns else [])
    + numeric_features
    + categorical_features
)

model_data = df[keep_columns].copy()


# ============================================================
# 4. HANDLE MISSING VALUES AND ENCODE CATEGORIES
# ============================================================

# Numeric missing values are replaced with their column median.
# Standardizing numeric variables improves numerical stability.
for col in numeric_features:
    model_data[col] = pd.to_numeric(
        model_data[col], errors="coerce"
    )

    median_value = model_data[col].median()

    if pd.isna(median_value):
        model_data.drop(columns=[col], inplace=True)
        numeric_features.remove(col)
        continue

    model_data[col] = model_data[col].fillna(median_value)

    standard_deviation = model_data[col].std()

    if pd.notna(standard_deviation) and standard_deviation > 0:
        model_data[col] = (
            model_data[col] - model_data[col].mean()
        ) / standard_deviation
    else:
        model_data[col] = 0.0

# Missing categories receive an explicit "Missing" label.
for col in categorical_features:
    model_data[col] = (
        model_data[col]
        .astype("string")
        .fillna("Missing")
        .astype(str)
    )

# One-hot encoding converts categories into binary indicator columns.
# drop_first=True avoids keeping a redundant reference category.
model_data = pd.get_dummies(
    model_data,
    columns=categorical_features,
    drop_first=True,
    dtype=float
)

# Remove predictor columns with no variation.
protected_columns = {
    "duration_days", "event", "household_id"
}

predictor_columns = [
    col for col in model_data.columns
    if col not in protected_columns
]

constant_columns = [
    col for col in predictor_columns
    if model_data[col].nunique(dropna=False) <= 1
]

model_data.drop(columns=constant_columns, inplace=True)

# Remove any remaining rows with invalid values in model variables.
model_data = model_data.replace([np.inf, -np.inf], np.nan)

columns_to_check = [
    col for col in model_data.columns
    if col != "household_id"
]

model_data = model_data.dropna(subset=columns_to_check).copy()

predictor_columns = [
    col for col in model_data.columns
    if col not in {"duration_days", "event", "household_id"}
]

if not predictor_columns:
    raise ValueError("No usable predictors remain after preprocessing.")

print("\nPredictors included in the model:")
for col in predictor_columns:
    print("-", col)

print("\nNumber of encoded predictor columns:", len(predictor_columns))


# ============================================================
# 5. FIT COX PROPORTIONAL HAZARDS MODEL
# ============================================================

cph = CoxPHFitter(penalizer=0.1)

# Customers may appear in more than one campaign.
# If household_id exists, use clustered standard errors to account
# for within-household dependence in the uncertainty estimates.
if "household_id" in model_data.columns:
    cph.fit(
        model_data,
        duration_col="duration_days",
        event_col="event",
        cluster_col="household_id",
        robust=True
    )
else:
    cph.fit(
        model_data,
        duration_col="duration_days",
        event_col="event"
    )

print("\nCox PH model fitted successfully.")
print(cph.summary)


# ============================================================
# 6. SAVE MODEL SUMMARY AND HAZARD RATIOS
# ============================================================

summary = cph.summary.reset_index()

# Rename the first column consistently; it contains predictor names.
summary = summary.rename(columns={summary.columns[0]: "predictor"})

summary.to_csv(
    OUTPUT_DIR / "cox_model_summary.csv",
    index=False
)

print("\nModel summary saved.")


# ============================================================
# 7. PLOT THE MOST STATISTICALLY ASSOCIATED PREDICTORS
# ============================================================

# Display up to 20 predictors with the smallest p-values.
plot_data = cph.summary.copy()
plot_data = plot_data.replace([np.inf, -np.inf], np.nan)
plot_data = plot_data.dropna(subset=["exp(coef)"])

plot_data = plot_data.sort_values("p").head(20)

if not plot_data.empty:
    plot_data = plot_data.sort_values("exp(coef)")

    hazard_ratios = plot_data["exp(coef)"]
    lower_ci = plot_data["exp(coef) lower 95%"]
    upper_ci = plot_data["exp(coef) upper 95%"]

    positions = np.arange(len(plot_data))

    plt.figure(figsize=(10, max(6, len(plot_data) * 0.35)))

    plt.errorbar(
        hazard_ratios,
        positions,
        xerr=[
            hazard_ratios - lower_ci,
            upper_ci - hazard_ratios
        ],
        fmt="o",
        capsize=3
    )

    plt.axvline(
        x=1,
        linestyle="--",
        linewidth=1
    )

    plt.xscale("log")
    plt.yticks(positions, plot_data.index)
    plt.xlabel("Hazard ratio (log scale)")
    plt.ylabel("Predictor")
    plt.title("Cox PH: Hazard Ratios and 95% Confidence Intervals")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "cox_hazard_ratios.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Hazard ratio plot saved.")


# ============================================================
# 8. CHECK THE PROPORTIONAL HAZARDS ASSUMPTION
# ============================================================

ph_output = OUTPUT_DIR / "ph_assumption_check.txt"

# The diagnostic routine needs the model predictors and outcome,
# not the household cluster ID.
diagnostic_data = model_data.drop(
    columns=["household_id"], errors="ignore"
).copy()

try:
    with open(ph_output, "w", encoding="utf-8") as file:
        file.write(
            "Proportional hazards assumption diagnostic\n"
            "==========================================\n"
        )
        file.write(
            "The test assesses whether predictor effects appear "
            "to change over time.\n\n"
        )

        cph.check_assumptions(
            diagnostic_data,
            p_value_threshold=0.05,
            show_plots=False
        )

        file.write(
            "Diagnostic completed. Review the console output and "
            "the model assumptions before interpreting hazard ratios.\n"
        )

    print("PH assumption diagnostic saved.")

except Exception as error:
    message = (
        "The model fitted, but the proportional hazards diagnostic "
        f"could not complete automatically.\nError: {error}\n"
        "Do not assume the PH assumption is satisfied. "
        "We can investigate this diagnostic separately."
    )

    with open(ph_output, "w", encoding="utf-8") as file:
        file.write(message)

    print("\n" + message)


# ============================================================
# 9. FINISH
# ============================================================

print("\nCox PH analysis completed.")
print("Results saved to:")
print(OUTPUT_DIR)
