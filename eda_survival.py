
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Paths and data loading
# --------------------------------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("eda_results")
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "model_dataset.csv"

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["start_date", "end_date", "redemption_date"]
)

print("=" * 60)
print("SURVIVAL ANALYSIS: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("Unique households:", df["household_id"].nunique())
print("Unique campaigns:", df["campaign_id"].nunique())

# --------------------------------------------------
# 2. Overall event and censoring summary
# --------------------------------------------------
event_summary = pd.DataFrame({
    "category": ["Coupon redemption", "Censored"],
    "count": [
        int((df["event"] == 1).sum()),
        int((df["event"] == 0).sum())
    ]
})

event_summary["percentage"] = (
    event_summary["count"] / len(df) * 100
).round(2)

event_summary.to_csv(
    OUTPUT_DIR / "event_summary.csv", index=False
)

print("\n--- Event and censoring summary ---")
print(event_summary.to_string(index=False))

# --------------------------------------------------
# 3. Duration summary by event status
# --------------------------------------------------
duration_summary = (
    df.groupby("event")["duration_days"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .round(2)
    .reset_index()
)

duration_summary["status"] = duration_summary["event"].map({
    0: "Censored",
    1: "Coupon redemption"
})

duration_summary.to_csv(
    OUTPUT_DIR / "duration_summary.csv", index=False
)

print("\n--- Recorded duration by event status ---")
print(duration_summary.to_string(index=False))

print(
    "\nZero-day observations:",
    int((df["duration_days"] == 0).sum())
)

print(
    "Zero-day redemption events:",
    int(((df["duration_days"] == 0) & (df["event"] == 1)).sum())
)

# --------------------------------------------------
# 4. Event rates by campaign type
# --------------------------------------------------
campaign_type_summary = (
    df.groupby("campaign_type", dropna=False)
    .agg(
        observations=("event", "size"),
        redemptions=("event", "sum"),
        median_recorded_duration=("duration_days", "median")
    )
    .reset_index()
)

campaign_type_summary["event_rate_percent"] = (
    campaign_type_summary["redemptions"]
    / campaign_type_summary["observations"] * 100
).round(2)

campaign_type_summary["median_recorded_duration"] = (
    campaign_type_summary["median_recorded_duration"].round(2)
)

campaign_type_summary.to_csv(
    OUTPUT_DIR / "campaign_type_summary.csv", index=False
)

print("\n--- Results by campaign type ---")
print(campaign_type_summary.to_string(index=False))

# --------------------------------------------------
# 5. Prior shopping behavior
# --------------------------------------------------
shopping_cols = [
    "prior_trip_count",
    "prior_total_spend",
    "prior_avg_basket_value",
    "days_since_last_purchase"
]

shopping_summary = (
    df[shopping_cols]
    .describe()
    .round(2)
    .T
)

shopping_summary.to_csv(
    OUTPUT_DIR / "shopping_behavior_summary.csv"
)

print("\n--- Prior shopping behavior ---")
print(shopping_summary.to_string())

# --------------------------------------------------
# 6. Demographic missingness by event status
# --------------------------------------------------
demographic_cols = [
    "age",
    "income",
    "home_ownership",
    "marital_status",
    "household_size",
    "household_comp",
    "kids_count"
]

missingness_by_event = (
    df.groupby("event")[demographic_cols]
    .agg(lambda column: column.isna().mean() * 100)
    .round(2)
    .T
)

missingness_by_event.columns = [
    "censored_missing_percent",
    "redemption_missing_percent"
]

missingness_by_event.to_csv(
    OUTPUT_DIR / "demographic_missingness_by_event.csv"
)

print("\n--- Demographic missingness by event status (%) ---")
print(missingness_by_event.to_string())

# --------------------------------------------------
# 7. Figure 1: Event and censoring counts
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

ax.bar(
    ["Coupon redemption", "Censored"],
    event_summary["count"]
)

ax.set_title("Coupon Redemption Events and Censored Observations")
ax.set_ylabel("Number of Household-Campaign Pairs")
ax.tick_params(axis="x", rotation=10)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "event_censoring_counts.png",
    dpi=300
)
plt.close(fig)

# --------------------------------------------------
# 8. Figure 2: Recorded duration distributions
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for event_value, label in [
    (1, "Coupon redemption"),
    (0, "Censored")
]:
    durations = df.loc[
        (df["event"] == event_value) &
        (df["duration_days"] > 0),
        "duration_days"
    ]

    ax.hist(
        durations,
        bins=25,
        alpha=0.55,
        label=label
    )

ax.set_title("Recorded Follow-up Duration by Event Status")
ax.set_xlabel("Duration (days)")
ax.set_ylabel("Number of Observations")
ax.legend()

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "duration_distributions.png",
    dpi=300
)
plt.close(fig)

# --------------------------------------------------
# 9. Figure 3: Event rate by campaign type
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

ax.bar(
    campaign_type_summary["campaign_type"].astype(str),
    campaign_type_summary["event_rate_percent"]
)

ax.set_title("Coupon Redemption Rate by Campaign Type")
ax.set_xlabel("Campaign Type")
ax.set_ylabel("Redemption Events (%)")

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "event_rate_by_campaign_type.png",
    dpi=300
)
plt.close(fig)

# --------------------------------------------------
# 10. Figure 4: Prior shopping behavior
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(
    df["prior_trip_count"],
    bins=30
)
axes[0].set_title("Prior Trip Count")
axes[0].set_xlabel("Trips before campaign")
axes[0].set_ylabel("Household-Campaign Pairs")

axes[1].hist(
    df["prior_total_spend"],
    bins=30
)
axes[1].set_title("Prior Total Spending")
axes[1].set_xlabel("Prior spending")
axes[1].set_ylabel("Household-Campaign Pairs")

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "prior_shopping_behavior.png",
    dpi=300
)
plt.close(fig)

# --------------------------------------------------
# 11. Final checks
# --------------------------------------------------
print("\n--- Final checks ---")
print("Missing event values:", int(df["event"].isna().sum()))
print("Missing duration values:", int(df["duration_days"].isna().sum()))
print("Duplicate household-campaign pairs:",
      int(df.duplicated(["household_id", "campaign_id"]).sum()))

print("\nEDA completed.")
print("Tables and figures saved in:", OUTPUT_DIR.resolve())
