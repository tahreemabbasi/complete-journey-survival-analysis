
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")

# --------------------------------------------------
# 1. Load the survival cohort and campaign metadata
# --------------------------------------------------

cohort = pd.read_csv(
    DATA_DIR / "survival_cohort.csv",
    parse_dates=["start_date", "end_date", "redemption_date"],
)

campaigns = pd.read_csv(
    DATA_DIR / "campaign_descriptions.csv",
    parse_dates=["start_date", "end_date"],
)

demographics = pd.read_csv(
    DATA_DIR / "demographics.csv",
)

# Keep campaigns completely within the transaction
# observation window.
observation_start = pd.Timestamp("2017-01-01")
observation_end = pd.Timestamp("2017-12-31")

eligible_campaigns = campaigns[
    (campaigns["start_date"] >= observation_start)
    & (campaigns["end_date"] <= observation_end)
].copy()

eligible_ids = set(eligible_campaigns["campaign_id"])

cohort = cohort[
    cohort["campaign_id"].isin(eligible_ids)
].copy()

print("Eligible campaigns:", len(eligible_ids))
print("Cohort pairs:", len(cohort))

expected_rows = len(cohort)

# --------------------------------------------------
# 2. Join campaign characteristics
# --------------------------------------------------

# Remove dates from the metadata copy because the cohort
# already contains the campaign dates.
campaign_features = campaigns[
    ["campaign_id", "campaign_type"]
].copy()

df = cohort.merge(
    campaign_features,
    on="campaign_id",
    how="left",
    validate="many_to_one",
)

# Campaign duration is a known campaign characteristic.
df["campaign_duration_days"] = (
    df["end_date"] - df["start_date"]
).dt.days

# --------------------------------------------------
# 3. Join household demographics
# --------------------------------------------------

df = df.merge(
    demographics,
    on="household_id",
    how="left",
    validate="many_to_one",
    indicator="_demographic_merge",
)

print(
    "Pairs without demographic records:",
    (df["_demographic_merge"] == "left_only").sum(),
)

df = df.drop(columns="_demographic_merge")

# --------------------------------------------------
# 4. Load transaction data
# --------------------------------------------------

transactions = pd.read_csv(
    DATA_DIR / "transactions.csv",
    usecols=[
        "household_id",
        "basket_id",
        "sales_value",
        "transaction_timestamp",
    ],
    parse_dates=["transaction_timestamp"],
)

transactions["transaction_date"] = (
    transactions["transaction_timestamp"].dt.normalize()
)

# --------------------------------------------------
# 5. Aggregate product lines into basket summaries
# --------------------------------------------------

# Each basket can contain multiple product-level rows.
# Count each basket once for trip count.
baskets = (
    transactions.groupby(
        ["household_id", "basket_id"],
        as_index=False,
    )
    .agg(
        basket_value=("sales_value", "sum"),
        basket_timestamp=("transaction_timestamp", "min"),
        basket_date=("transaction_date", "min"),
    )
)

# We no longer need the line-level transaction table.
del transactions

print("Unique household baskets:", len(baskets))

# --------------------------------------------------
# 6. Create pre-campaign features
# --------------------------------------------------

# Calculate features separately for each household-campaign
# pair, using only baskets strictly before campaign start.
feature_records = []

for row in df[
    ["household_id", "campaign_id", "start_date"]
].itertuples(index=False):

    household_id = row.household_id
    campaign_id = row.campaign_id
    start_date = row.start_date

    history = baskets[
        (baskets["household_id"] == household_id)
        & (baskets["basket_date"] < start_date)
    ]

    if history.empty:
        feature_records.append({
            "household_id": household_id,
            "campaign_id": campaign_id,
            "prior_trip_count": 0,
            "prior_total_spend": 0.0,
            "prior_avg_basket_value": np.nan,
            "days_since_last_purchase": np.nan,
        })
        continue

    last_purchase = history["basket_date"].max()

    feature_records.append({
        "household_id": household_id,
        "campaign_id": campaign_id,
        "prior_trip_count": history["basket_id"].nunique(),
        "prior_total_spend": history["basket_value"].sum(),
        "prior_avg_basket_value": history["basket_value"].mean(),
        "days_since_last_purchase": (
            start_date - last_purchase
        ).days,
    })

history_features = pd.DataFrame(feature_records)

# --------------------------------------------------
# 7. Join historical behavior features
# --------------------------------------------------

df = df.merge(
    history_features,
    on=["household_id", "campaign_id"],
    how="left",
    validate="one_to_one",
)

# --------------------------------------------------
# 8. Validate the analytical dataset
# --------------------------------------------------

print("\n--- FINAL VALIDATION ---")
print("Expected rows:", expected_rows)
print("Actual rows:", len(df))

print(
    "Duplicate household-campaign pairs:",
    df.duplicated(
        ["household_id", "campaign_id"]
    ).sum(),
)

print(
    "Missing event values:",
    df["event"].isna().sum(),
)

print(
    "Missing duration values:",
    df["duration_days"].isna().sum(),
)

print("\nMissing values by column:")
print(
    df.isna().sum()
    .sort_values(ascending=False)
    .to_string()
)

assert len(df) == expected_rows, (
    "Row count changed unexpectedly."
)

assert not df.duplicated(
    ["household_id", "campaign_id"]
).any(), "Duplicate cohort pairs found."

# --------------------------------------------------
# 9. Save
# --------------------------------------------------

output_path = DATA_DIR / "analytical_dataset.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved analytical dataset to: {output_path}")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head().to_string(index=False))
