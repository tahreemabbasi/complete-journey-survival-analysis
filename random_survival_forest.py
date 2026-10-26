
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored


# ============================================================
# 1. FILE PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_FILE = PROJECT_DIR / "data" / "model_dataset.csv"

OUTPUT_DIR = (
    PROJECT_DIR / "model_results" / "random_survival_forest"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

required_columns = ["duration_days", "event", "household_id"]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df["duration_days"] = pd.to_numeric(
    df["duration_days"], errors="coerce"
)
df["event"] = pd.to_numeric(
    df["event"], errors="coerce"
)

df = df.dropna(
    subset=["duration_days", "event", "household_id"]
).copy()

df = df[df["event"].isin([0, 1])].copy()
df = df[df["duration_days"] >= 0].copy()
df["event"] = df["event"].astype(int)

# This first model excludes zero-day durations because the
# survival modelling workflow requires positive durations.
zero_day_rows = int((df["duration_days"] == 0).sum())
zero_day_events = int(
    (
        (df["duration_days"] == 0)
        & (df["event"] == 1)
    ).sum()
)

df = df[df["duration_days"] > 0].copy()

if df.empty or df["event"].sum() == 0:
    raise ValueError(
        "No valid positive-duration observations with events remain."
    )

print("Valid positive-duration observations:", len(df))
print("Redemption events:", int(df["event"].sum()))
print("Censored observations:", int((df["event"] == 0).sum()))
print("Zero-day rows excluded:", zero_day_rows)
print("Zero-day redemption events excluded:", zero_day_events)


# ============================================================
# 3. SELECT PREDICTORS
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
    col for col in numeric_candidates if col in df.columns
]

categorical_features = [
    col for col in categorical_candidates if col in df.columns
]

if not numeric_features and not categorical_features:
    raise ValueError("No expected predictor columns were found.")

feature_columns = numeric_features + categorical_features

if not feature_columns:
    raise ValueError("No usable predictors were found.")

# Convert numeric candidates safely.
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert categories to strings so values such as "5+" are handled
# as categories rather than incorrectly treated as numbers.
for col in categorical_features:
    df[col] = (
        df[col]
        .astype("string")
        .fillna("Missing")
        .astype(str)
    )

X = df[feature_columns].copy()

# Survival outcome: event=True means coupon redemption was observed.
y = Surv.from_arrays(
    event=df["event"].astype(bool).to_numpy(),
    time=df["duration_days"].to_numpy(dtype=float)
)

# Use household IDs to keep each household in only one split.
groups = df["household_id"].astype(str).to_numpy()


# ============================================================
# 4. SPLIT INTO TRAINING AND TEST SETS BY HOUSEHOLD
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y[train_idx]
y_test = y[test_idx]

train_events = int(y_train["event"].sum())
test_events = int(y_test["event"].sum())

if train_events == 0 or test_events == 0:
    raise ValueError(
        "The train/test split does not contain redemption events "
        "in both sets. The split needs to be reconsidered."
    )

train_households = set(groups[train_idx])
test_households = set(groups[test_idx])

overlap = train_households.intersection(test_households)

if overlap:
    raise RuntimeError(
        "Household leakage detected between training and testing."
    )

print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Training redemption events:", train_events)
print("Testing redemption events:", test_events)
print("Households in training:", len(train_households))
print("Households in testing:", len(test_households))
print("Household overlap:", len(overlap))


# ============================================================
# 5. PREPROCESS THE PREDICTORS
# ============================================================

# Preprocessing is fitted on training data only.
# The fitted transformations are then applied to test data.

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median",
            add_indicator=True
        )
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="Missing"
        )
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# ============================================================
# 6. BUILD THE RANDOM SURVIVAL FOREST
# ============================================================

rsf = RandomSurvivalForest(
    n_estimators=300,
    min_samples_split=10,
    min_samples_leaf=15,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

model = Pipeline([
    ("preprocessing", preprocessor),
    ("random_survival_forest", rsf)
])


# ============================================================
# 7. TRAIN THE MODEL
# ============================================================

print("\nTraining Random Survival Forest...")

model.fit(X_train, y_train)

print("Model training completed.")


# ============================================================
# 8. EVALUATE ON UNSEEN HOUSEHOLDS
# ============================================================

# The model returns risk scores. Higher scores mean higher
# predicted event risk, not a predicted number of days.

risk_scores = model.predict(X_test)

c_index = concordance_index_censored(
    y_test["event"],
    y_test["time"],
    risk_scores
)[0]

print("\nTest-set concordance index:", round(float(c_index), 4))

metrics = pd.DataFrame([{
    "model": "Random Survival Forest",
    "training_observations": len(X_train),
    "testing_observations": len(X_test),
    "training_events": train_events,
    "testing_events": test_events,
    "test_c_index": float(c_index),
    "zero_day_rows_excluded": zero_day_rows,
    "zero_day_events_excluded": zero_day_events
}])

metrics.to_csv(
    OUTPUT_DIR / "model_metrics.csv",
    index=False
)


# ============================================================
# 9. SAVE TEST-SET PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "household_id": df.iloc[test_idx]["household_id"].to_numpy(),
    "duration_days": y_test["time"],
    "event_observed": y_test["event"].astype(int),
    "predicted_risk_score": risk_scores
})

predictions.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)


# ============================================================
# 10. PLOT THE TEST-SET RISK SCORES
# ============================================================

plt.figure(figsize=(8, 6))

plt.boxplot(
    [
        risk_scores[y_test["event"]],
        risk_scores[~y_test["event"]]
    ],
    tick_labels=[
        "Redemption observed",
        "Censored"
    ]
)

plt.ylabel("Predicted risk score")
plt.title("RSF Risk Scores on the Test Set")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "test_risk_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. SAVE A RUN SUMMARY
# ============================================================

summary_text = (
    "Random Survival Forest Analysis\n"
    "===============================\n"
    f"Positive-duration observations: {len(df)}\n"
    f"Training observations: {len(X_train)}\n"
    f"Testing observations: {len(X_test)}\n"
    f"Training events: {train_events}\n"
    f"Testing events: {test_events}\n"
    f"Zero-day rows excluded: {zero_day_rows}\n"
    f"Zero-day redemption events excluded: {zero_day_events}\n"
    f"Test-set C-index: {c_index:.4f}\n\n"
    "The C-index measures ranking ability, not exact prediction "
    "of redemption dates. The model does not establish causation.\n"
)

with open(
    OUTPUT_DIR / "run_summary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(summary_text)

print("\nAnalysis completed successfully.")
print("Results saved to:")
print(OUTPUT_DIR)
