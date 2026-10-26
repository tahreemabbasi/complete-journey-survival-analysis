
from pathlib import Path
import pandas as pd

DATA = Path("data")

# Load the existing coupon-redemption cohort
cohort = pd.read_csv(
    DATA / "survival_cohort.csv",
    parse_dates=["start_date", "end_date", "redemption_date"]
)

# Load campaign dates and transaction observation window
campaigns = pd.read_csv(
    DATA / "campaign_descriptions.csv",
    parse_dates=["start_date", "end_date"]
)

tx = pd.read_csv(
    DATA / "transactions.csv",
    usecols=["household_id", "transaction_timestamp"],
    parse_dates=["transaction_timestamp"]
)

tx["transaction_date"] = tx["transaction_timestamp"].dt.normalize()

data_start = tx["transaction_date"].min()
data_end = tx["transaction_date"].max()

# Keep campaigns fully covered by transaction records
eligible = campaigns[
    (campaigns["start_date"] >= data_start)
    & (campaigns["end_date"] <= data_end)
].copy()

print("Eligible campaigns:", len(eligible))

# Restrict both formulations to the same cohort
cohort = cohort[cohort["campaign_id"].isin(eligible["campaign_id"])].copy()

# Formulation A: time to first qualifying coupon redemption
cohort["event_A"] = (
    cohort["redemption_date"].notna()
    & (cohort["redemption_date"] >= cohort["start_date"])
    & (cohort["redemption_date"] <= cohort["end_date"])
).astype(int)

cohort["duration_A"] = (
    cohort["end_date"] - cohort["start_date"]
).dt.days

has_redemption = cohort["event_A"].eq(1)
cohort.loc[has_redemption, "duration_A"] = (
    cohort.loc[has_redemption, "redemption_date"]
    - cohort.loc[has_redemption, "start_date"]
).dt.days

# Reduce transactions to unique household-day combinations
tx_daily = tx[["household_id", "transaction_date"]].drop_duplicates()

# Formulation B: time to first purchase on/after campaign start
purchase_records = []

for campaign_id, group in cohort.groupby("campaign_id"):
    start = group["start_date"].iloc[0]
    end = group["end_date"].iloc[0]
    households = group["household_id"].unique()

    relevant = tx_daily[
        tx_daily["household_id"].isin(households)
        & (tx_daily["transaction_date"] >= start)
        & (tx_daily["transaction_date"] <= end)
    ]

    first = (
        relevant.sort_values("transaction_date")
        .drop_duplicates("household_id")
        [["household_id", "transaction_date"]]
        .rename(columns={"transaction_date": "first_purchase_date"})
    )

    first["campaign_id"] = campaign_id
    purchase_records.append(first)

purchases = pd.concat(purchase_records, ignore_index=True)

cohort = cohort.merge(
    purchases,
    on=["household_id", "campaign_id"],
    how="left",
    validate="one_to_one"
)

cohort["event_B"] = cohort["first_purchase_date"].notna().astype(int)
cohort["duration_B"] = (
    cohort["end_date"] - cohort["start_date"]
).dt.days

has_purchase = cohort["event_B"].eq(1)
cohort.loc[has_purchase, "duration_B"] = (
    cohort.loc[has_purchase, "first_purchase_date"]
    - cohort.loc[has_purchase, "start_date"]
).dt.days

# Print comparable summary statistics
print("\nCOMPARISON: SAME CAMPAIGNS AND HOUSEHOLD- CAMPAIGN PAIRS")
print("Household-campaign pairs:", len(cohort))

for name, event_col, duration_col in [
    ("A: Coupon redemption", "event_A", "duration_A"),
    ("B: Any purchase", "event_B", "duration_B"),
]:
    events = int(cohort[event_col].sum())
    n = len(cohort)

    print(f"\n{name}")
    print("Events:", events)
    print("Censored:", n - events)
    print("Event rate (%):", round(100 * events / n, 2))
    print("Censoring rate (%):", round(100 * (n - events) / n, 2))
    print("Median observed duration (days):",
          round(cohort[duration_col].median(), 2))
    print("Zero-day events:",
          int(((cohort[event_col] == 1)
               & (cohort[duration_col] == 0)).sum()))

cohort.to_csv(DATA / "formulation_comparison.csv", index=False)
print("\nSaved: data/formulation_comparison.csv")
