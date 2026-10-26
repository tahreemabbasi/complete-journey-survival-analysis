
from pathlib import Path
import pandas as pd

DATA = Path("data")

assignments = pd.read_csv(DATA / "campaigns.csv")
descriptions = pd.read_csv(DATA / "campaign_descriptions.csv")
redemptions = pd.read_csv(DATA / "coupon_redemptions.csv")
coupons = pd.read_csv(DATA / "coupons.csv")

# One row per household-campaign assignment
cohort = assignments.drop_duplicates(
    ["household_id", "campaign_id"]
).merge(
    descriptions[["campaign_id", "start_date", "end_date"]],
    on="campaign_id",
    how="left",
    validate="many_to_one"
)

for col in ["start_date", "end_date"]:
    cohort[col] = pd.to_datetime(cohort[col])

redemptions["redemption_date"] = pd.to_datetime(
    redemptions["redemption_date"]
)

# Check whether each redemption's coupon belongs to its campaign.
valid_coupons = coupons[
    ["coupon_upc", "campaign_id"]
].drop_duplicates()

checked = redemptions.merge(
    valid_coupons,
    on=["coupon_upc", "campaign_id"],
    how="left",
    indicator=True
)

coupon_match = checked["_merge"].eq("both")
print("Redemption records:", len(redemptions))
print("Records matching a campaign coupon:", int(coupon_match.sum()))
print("Records not matching a campaign coupon:", int((~coupon_match).sum()))

# Keep redemptions linked to a known campaign coupon.
valid_redemptions = checked.loc[
    coupon_match,
    ["household_id", "campaign_id", "redemption_date"]
].copy()

# Check whether the household was assigned to that campaign.
assignment_keys = cohort[["household_id", "campaign_id"]].drop_duplicates()

valid_redemptions = valid_redemptions.merge(
    assignment_keys,
    on=["household_id", "campaign_id"],
    how="left",
    indicator=True
)

assigned = valid_redemptions["_merge"].eq("both")
print("Redemptions from assigned household-campaign pairs:", int(assigned.sum()))
print("Redemptions from unassigned pairs:", int((~assigned).sum()))

valid_redemptions = valid_redemptions.loc[assigned].drop(
    columns="_merge"
)

# First qualifying redemption per household-campaign pair.
first_redemption = (
    valid_redemptions.groupby(
        ["household_id", "campaign_id"],
        as_index=False
    )["redemption_date"]
    .min()
)

cohort = cohort.merge(
    first_redemption,
    on=["household_id", "campaign_id"],
    how="left",
    validate="one_to_one"
)

# Event = at least one qualifying redemption during the campaign.
cohort["event"] = (
    cohort["redemption_date"].notna()
    & (cohort["redemption_date"] >= cohort["start_date"])
    & (cohort["redemption_date"] <= cohort["end_date"])
).astype(int)

# Observed time: first redemption for events; campaign end for non-events.
cohort["observed_end"] = cohort["end_date"]
cohort.loc[cohort["event"].eq(1), "observed_end"] = cohort.loc[
    cohort["event"].eq(1), "redemption_date"
]

cohort["duration_days"] = (
    cohort["observed_end"] - cohort["start_date"]
).dt.days

print("\nSURVIVAL COHORT SUMMARY")
print("Household-campaign pairs:", len(cohort))
print("Events:", int(cohort["event"].sum()))
print("No observed event / censored:", int((cohort["event"] == 0).sum()))
print("Event rate (%):", round(cohort["event"].mean() * 100, 2))
print("Censoring rate (%):", round((cohort["event"] == 0).mean() * 100, 2))
print("Duration summary (days):")
print(cohort["duration_days"].describe())

print("\nEVENTS BY CAMPAIGN TYPE")
print(
    cohort.merge(
        descriptions[["campaign_id", "campaign_type"]],
        on="campaign_id",
        validate="many_to_one"
    ).groupby("campaign_type")["event"].agg(
        pairs="size", events="sum", event_rate="mean"
    ).assign(event_rate_pct=lambda x: (x["event_rate"] * 100).round(2))
)

cohort.to_csv(DATA / "survival_cohort.csv", index=False)
print("\nSaved:", DATA / "survival_cohort.csv")
