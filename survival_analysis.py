
from pathlib import Path
import warnings
from contextlib import redirect_stdout
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from lifelines import CoxPHFitter
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
)


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "model_dataset.csv"

OUTPUT_DIR = PROJECT_DIR / "model_results" / "model_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25
AUC_HORIZONS = [7, 14, 30]


# ============================================================
# 2. LOAD AND VALIDATE THE COHORT
# ============================================================

df = pd.read_csv(DATA_FILE)

required = ["duration_days", "event", "household_id"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Required columns missing: {missing}")

df["duration_days"] = pd.to_numeric(
    df["duration_days"], errors="coerce"
)
df["event"] = pd.to_numeric(df["event"], errors="coerce")

df = df.dropna(
    subset=["duration_days", "event", "household_id"]
).copy()

df = df[df["event"].isin([0, 1])].copy()
df = df[df["duration_days"] >= 0].copy()
df["event"] = df["event"].astype(int)

# Record zero-day observations before excluding them.
zero_day_rows = int((df["duration_days"] == 0).sum())
zero_day_events = int(
    (
        (df["duration_days"] == 0)
        & (df["event"] == 1)
    ).sum()
)

# Cox PH and RSF will use the same positive-duration cohort.
df = df[df["duration_days"] > 0].copy()

if df.empty or df["event"].sum() == 0:
    raise ValueError(
        "No valid positive-duration observations with events."
    )


# ============================================================
# 3. SELECT THE SAME PREDICTORS FOR BOTH MODELS
# ============================================================

numeric_candidates = [
    "prior_trip_count",
    "prior_total_spend",
    "prior_avg_basket_value",
    "days_since_last_purchase",
    "kids_count",
    "demographics_missing",
]

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
    c for c in numeric_candidates
    if c in df.columns
]

categorical_features = [
    c for c in categorical_candidates
    if c in df.columns
]

if not numeric_features and not categorical_features:
    raise ValueError("No candidate predictors were found.")

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop numeric variables that have no usable values.
numeric_features = [
    c for c in numeric_features
    if df[c].notna().any()
]

for col in categorical_features:
    df[col] = (
        df[col]
        .astype("string")
        .fillna("Missing")
        .astype(str)
    )

feature_columns = numeric_features + categorical_features

X = df[feature_columns].copy()
groups = df["household_id"].astype(str).to_numpy()

y = Surv.from_arrays(
    event=df["event"].astype(bool).to_numpy(),
    time=df["duration_days"].to_numpy(dtype=float),
)


# ============================================================
# 4. CREATE ONE HOUSEHOLD-GROUPED TRAIN/TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train_raw = X.iloc[train_idx].copy()
X_test_raw = X.iloc[test_idx].copy()

y_train = y[train_idx]
y_test = y[test_idx]

train_groups = groups[train_idx]
test_groups = groups[test_idx]

if set(train_groups).intersection(set(test_groups)):
    raise RuntimeError("Household overlap detected.")

if y_train["event"].sum() == 0 or y_test["event"].sum() == 0:
    raise ValueError(
        "Both training and test sets must contain observed events."
    )

print("Positive-duration cohort:", len(df))
print("Zero-day rows excluded:", zero_day_rows)
print("Zero-day redemption events excluded:", zero_day_events)
print("Training rows:", len(train_idx))
print("Test rows:", len(test_idx))
print("Training events:", int(y_train["event"].sum()))
print("Test events:", int(y_test["event"].sum()))
print("Household overlap:", len(
    set(train_groups).intersection(set(test_groups))
))


# ============================================================
# 5. FIT PREPROCESSING ON TRAINING DATA ONLY
# ============================================================

transformers = []

if numeric_features:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("scaler", StandardScaler()),
    ])

    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Missing"
        )),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False
        )),
    ])

    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
)

train_array = preprocessor.fit_transform(X_train_raw)
test_array = preprocessor.transform(X_test_raw)

encoded_columns = preprocessor.get_feature_names_out()

X_train = pd.DataFrame(
    train_array,
    columns=encoded_columns,
    index=X_train_raw.index,
)

X_test = pd.DataFrame(
    test_array,
    columns=encoded_columns,
    index=X_test_raw.index,
)

# Remove columns that have no variation in training data.
constant_columns = [
    col for col in X_train.columns
    if X_train[col].nunique(dropna=False) <= 1
]

X_train.drop(columns=constant_columns, inplace=True)
X_test.drop(columns=constant_columns, inplace=True)

if X_train.shape[1] == 0:
    raise ValueError("No usable predictors remain after preprocessing.")

# Use identical predictor columns in both models.
X_test = X_test[X_train.columns]

print("Encoded predictors used:", X_train.shape[1])


# ============================================================
# 6. FIT COX PROPORTIONAL HAZARDS MODEL
# ============================================================

cox_train = X_train.copy()
cox_train["duration_days"] = y_train["time"]
cox_train["event"] = y_train["event"].astype(int)
cox_train["household_id"] = train_groups

cox = CoxPHFitter(penalizer=0.1)

cox.fit(
    cox_train,
    duration_col="duration_days",
    event_col="event",
    cluster_col="household_id",
    robust=True,
)

cox_risk = cox.predict_partial_hazard(X_test).to_numpy().ravel()

cox_cindex = concordance_index_censored(
    y_test["event"],
    y_test["time"],
    cox_risk,
)[0]

print("\nCox PH test C-index:", round(float(cox_cindex), 4))

cox_summary = cox.summary.reset_index()
cox_summary = cox_summary.rename(
    columns={cox_summary.columns[0]: "predictor"}
)

cox_summary.to_csv(
    OUTPUT_DIR / "cox_model_summary.csv",
    index=False,
)


# ============================================================
# 7. CHECK THE COX PROPORTIONAL HAZARDS ASSUMPTION
# ============================================================

diagnostic_file = OUTPUT_DIR / "cox_ph_diagnostics.txt"
diagnostic_output = io.StringIO()

try:
    # Use the same training dataframe structure used to fit Cox.
    # household_id is retained for clustering, not as a predictor.
    diagnostic_data = cox_train.copy()

    with redirect_stdout(diagnostic_output):
        cox.check_assumptions(
            diagnostic_data,
            p_value_threshold=0.05,
            show_plots=False,
        )

    diagnostic_text = diagnostic_output.getvalue()

    if not diagnostic_text.strip():
        diagnostic_text = (
            "The diagnostic completed without printed output. "
            "This alone does not establish that the PH assumption holds."
        )

except Exception as exc:
    diagnostic_text = (
        "Cox PH diagnostic could not complete.\n"
        f"Error type: {type(exc).__name__}\n"
        f"Error: {exc}\n"
        "The proportional-hazards assumption remains unverified."
    )

with open(diagnostic_file, "w", encoding="utf-8") as f:
    f.write(diagnostic_text)

print("Cox PH diagnostic saved.")
print(diagnostic_text)

# ============================================================
# 8. FIT RANDOM SURVIVAL FOREST ON THE SAME SPLIT
# ============================================================

rsf = RandomSurvivalForest(
    n_estimators=300,
    min_samples_split=10,
    min_samples_leaf=15,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

rsf.fit(X_train, y_train)

rsf_risk = rsf.predict(X_test)

rsf_cindex = concordance_index_censored(
    y_test["event"],
    y_test["time"],
    rsf_risk,
)[0]

print("RSF test C-index:", round(float(rsf_cindex), 4))


# ============================================================
# 9. SAVE A FAIR C-INDEX COMPARISON
# ============================================================

comparison = pd.DataFrame([
    {
        "model": "Cox Proportional Hazards",
        "test_c_index": float(cox_cindex),
    },
    {
        "model": "Random Survival Forest",
        "test_c_index": float(rsf_cindex),
    },
])

comparison.to_csv(
    OUTPUT_DIR / "model_comparison_cindex.csv",
    index=False,
)

plt.figure(figsize=(8, 5))
plt.bar(
    comparison["model"],
    comparison["test_c_index"],
)
plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
    label="0.5 reference",
)
plt.ylim(0, 1)
plt.ylabel("Test-set C-index")
plt.title("Cox PH vs Random Survival Forest")
plt.xticks(rotation=10, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "model_comparison_cindex.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 10. SAVE TEST-SET RISK SCORES
# ============================================================

test_predictions = pd.DataFrame({
    "household_id": df.iloc[test_idx]["household_id"].to_numpy(),
    "duration_days": y_test["time"],
    "event_observed": y_test["event"].astype(int),
    "cox_risk_score": cox_risk,
    "rsf_risk_score": rsf_risk,
})

test_predictions.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
)


# ============================================================
# SECTION 11: TIME-DEPENDENT AUC
# ============================================================

from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.metrics import cumulative_dynamic_auc

# Estimate the censoring distribution using training data only
censoring_model = CensoringDistributionEstimator().fit(y_train)

# Check which test follow-up times are supported by the
# training censoring distribution
unique_test_times = np.unique(y_test["time"])

censoring_survival = {
    float(t): float(
        censoring_model.predict_proba(np.array([t], dtype=float))[0]
    )
    for t in unique_test_times
}

# Keep observations with positive censoring survival probability
auc_mask = np.array([
    censoring_survival[float(t)] > 0
    for t in y_test["time"]
])

y_test_auc = y_test[auc_mask]
cox_risk_auc = cox_risk[auc_mask]
rsf_risk_auc = rsf_risk[auc_mask]

print("\nTIME-DEPENDENT AUC EVALUATION SUPPORT")
print("Original test observations:", len(y_test))
print("Observations retained for AUC:", len(y_test_auc))
print("Observations excluded for unsupported follow-up:",
      int((~auc_mask).sum()))
print("Events retained for AUC:",
      int(y_test_auc["event"].sum()))

# Evaluate both models on the same eligible test observations
auc_results = []

for model_name, risk_scores in [
    ("Cox Proportional Hazards", cox_risk_auc),
    ("Random Survival Forest", rsf_risk_auc),
]:
    for horizon in [7, 14, 30]:

        try:
            if not (
                np.min(y_test_auc["time"])
                < horizon
                < np.max(y_test_auc["time"])
            ):
                raise ValueError(
                    "Horizon falls outside the evaluable test follow-up range."
                )

            auc_values, mean_auc = cumulative_dynamic_auc(
                y_train,
                y_test_auc,
                risk_scores,
                np.array([horizon], dtype=float),
            )

            auc_results.append({
                "model": model_name,
                "horizon_days": horizon,
                "time_dependent_auc": float(auc_values[0]),
                "mean_auc": float(mean_auc),
                "status": "OK",
            })

        except (ValueError, IndexError) as exc:
            auc_results.append({
                "model": model_name,
                "horizon_days": horizon,
                "time_dependent_auc": np.nan,
                "mean_auc": np.nan,
                "status": f"Not estimable: {exc}",
            })

auc_df = pd.DataFrame(auc_results)

auc_df.to_csv(
    OUTPUT_DIR / "time_dependent_auc.csv",
    index=False,
)

print("\nTIME-DEPENDENT AUC RESULTS")
print(auc_df.to_string(index=False))

# ============================================================
# 12. SAVE EVALUATION AUDIT AND SUMMARY
# ============================================================

audit = (
    "Fair Model Evaluation Audit\n"
    "===========================\n"
    f"Input file: {DATA_FILE}\n"
    f"Positive-duration observations: {len(df)}\n"
    f"Zero-day rows excluded: {zero_day_rows}\n"
    f"Zero-day redemption events excluded: {zero_day_events}\n"
    f"Training observations: {len(train_idx)}\n"
    f"Test observations: {len(test_idx)}\n"
    f"Training redemption events: {int(y_train['event'].sum())}\n"
    f"Test redemption events: {int(y_test['event'].sum())}\n"
    f"Household overlap: "
    f"{len(set(train_groups).intersection(set(test_groups)))}\n"
    f"Encoded predictors: {X_train.shape[1]}\n"
    f"Cox PH test C-index: {cox_cindex:.4f}\n"
    f"RSF test C-index: {rsf_cindex:.4f}\n\n"
    "Both models used the same cohort, household-grouped split, "
    "preprocessed predictors, and test observations.\n"
    "C-index measures ranking ability, not exact date prediction.\n"
    "Time-dependent AUC estimates and any limitations are recorded "
    "in time_dependent_auc.csv.\n"
)

with open(
    OUTPUT_DIR / "evaluation_audit.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(audit)

print("\nTime-dependent AUC results:")
print(pd.DataFrame(auc_results).to_string(index=False))

print("\nEvaluation completed.")
print("Results saved to:")
print(OUTPUT_DIR)
