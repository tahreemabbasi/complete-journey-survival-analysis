
from pathlib import Path
import pandas as pd

DATA = Path("data")

# Load only the tables needed for this validation
assignments = pd.read_csv(DATA / "campaigns.csv")
descriptions = pd.read_csv(DATA / "campaign_descriptions.csv")
redemptions = pd.read_csv(DATA / "coupon_redemptions.csv")

# Parse dates
for df, column in [
    (descriptions, "start_date"),
    (descriptions, "end_date"),
    (redemptions, "redemption_date"),
]:
    df[column] = pd.to_datetime(df[column], errors="coerce")

print("CAMPAIGN DATE RANGE")
print("Start:", descriptions["start_date"].min())
print("End:  ", descriptions["end_date"].max())

print("\nTABLE SIZES")
print("Household-campaign assignments:", len(assignments))
print("Campaign descriptions:", len(descriptions))
print("Coupon redemption records:", len(redemptions))

# Check campaign IDs missing from the descriptions table
print("\nMISSING CAMPAIGN DESCRIPTIONS")
print(
    assignments["campaign_id"].nunique()
    - assignments["campaign_id"].isin(descriptions["campaign_id"]).sum()
    if False else
    assignments.loc[
        ~assignments["campaign_id"].isin(descriptions["campaign_id"]),
        "campaign_id"
    ].nunique()
)

# Attach campaign dates to redemption records
r = redemptions.merge(
    descriptions[["campaign_id", "start_date", "end_date"]],
    on="campaign_id",
    how="left",
    validate="many_to_one"
)

valid = r["start_date"].notna() & r["end_date"].notna()

before = valid & (r["redemption_date"] < r["start_date"])
during = valid & (
    (r["redemption_date"] >= r["start_date"]) &
    (r["redemption_date"] <= r["end_date"])
)
after = valid & (r["redemption_date"] > r["end_date"])

print("\nREDEMPTION DATE CHECK")
print("Records without campaign dates:", int((~valid).sum()))
print("Before campaign start:", int(before.sum()))
print("Within campaign dates:", int(during.sum()))
print("After campaign end:", int(after.sum()))

print("\nEXAMPLES OF REDEMPTIONS BEFORE START")
print(
    r.loc[before, [
        "household_id", "campaign_id", "redemption_date",
        "start_date", "end_date"
    ]].head(10).to_string(index=False)
)

print("\nEXAMPLES OF REDEMPTIONS AFTER END")
print(
    r.loc[after, [
        "household_id", "campaign_id", "redemption_date",
        "start_date", "end_date"
    ]].head(10).to_string(index=False)
)
