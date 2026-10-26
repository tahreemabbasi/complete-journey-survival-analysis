
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
INPUT = DATA_DIR / "analytical_dataset.csv"

# 1. Load the existing analytical dataset
df = pd.read_csv(
    INPUT,
    parse_dates=[
        "start_date",
        "end_date",
        "redemption_date",
        "observed_end",
    ],
)

print("Initial shape:", df.shape)

# 2. Audit missingness BEFORE any treatment
audit = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2),
})

audit.to_csv(DATA_DIR / "missingness_audit.csv")
print("\nMissingness audit:")
print(audit.sort_values("missing_percent", ascending=False))

# 3. Add an indicator for unavailable demographic records
demographic_cols = [
    "age",
    "income",
    "home_ownership",
    "marital_status",
    "household_size",
    "household_comp",
    "kids_count",
]

df["demographics_missing"] = (
    df[demographic_cols].isna().all(axis=1).astype(int)
)

# 4. Handle households with no observed prior purchase
df["has_prior_purchase"] = (
    df["prior_trip_count"] > 0
).astype(int)

# No previous baskets means observed prior trips and spending are zero.
no_history = df["has_prior_purchase"] == 0

df.loc[no_history, "prior_trip_count"] = 0
df.loc[no_history, "prior_total_spend"] = 0.0

# Average basket value and recency are undefined without history.
# Preserve their missing values here; handle them in model preprocessing.
print(
    "\nPairs without prior purchase history:",
    int(no_history.sum()),
)

# 5. Basic validity checks
assert df["event"].isin([0, 1]).all()
assert df["duration_days"].notna().all()
assert (df["duration_days"] >= 0).all()

assert not df.duplicated(
    ["household_id", "campaign_id"]
).any()

print("\nEvent rate:", round(df["event"].mean() * 100, 2), "%")
print("Zero-day durations:", int((df["duration_days"] == 0).sum()))

# 6. Save the prepared dataset
output = DATA_DIR / "model_dataset.csv"
df.to_csv(output, index=False)

print("\nSaved:", output)
print("Final shape:", df.shape)
print("Preparation completed.")
