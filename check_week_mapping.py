
from pathlib import Path
import pandas as pd

DATA = Path("data")

# Read only the fields needed; parse timestamps as dates.
tx = pd.read_csv(
    DATA / "transactions.csv",
    usecols=["week", "transaction_timestamp"],
    parse_dates=["transaction_timestamp"]
)

promotions = pd.read_csv(
    DATA / "promotions.csv",
    usecols=["week"]
)

campaigns = pd.read_csv(
    DATA / "campaign_descriptions.csv",
    parse_dates=["start_date", "end_date"]
)

print("TRANSACTION WEEK RANGE")
print("Minimum week:", tx["week"].min())
print("Maximum week:", tx["week"].max())
print("Number of distinct weeks:", tx["week"].nunique())

print("\nTRANSACTION DATES BY WEEK")
print(
    tx.groupby("week")["transaction_timestamp"]
      .agg(first_date="min", last_date="max")
      .head(10)
      .to_string()
)

print("\nPROMOTION WEEK RANGE")
print("Minimum week:", promotions["week"].min())
print("Maximum week:", promotions["week"].max())
print("Number of distinct weeks:", promotions["week"].nunique())

print("\nCAMPAIGN DATES")
print(
    campaigns[["campaign_id", "start_date", "end_date"]]
    .sort_values("start_date")
    .to_string(index=False)
)
