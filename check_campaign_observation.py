
from pathlib import Path
import pandas as pd

DATA = Path("data")

tx = pd.read_csv(
    DATA / "transactions.csv",
    usecols=["transaction_timestamp"],
    parse_dates=["transaction_timestamp"]
)

campaigns = pd.read_csv(
    DATA / "campaign_descriptions.csv",
    parse_dates=["start_date", "end_date"]
)

data_start = tx["transaction_timestamp"].min().normalize()
data_end = tx["transaction_timestamp"].max().normalize()

campaigns["fully_observed"] = (
    (campaigns["start_date"] >= data_start)
    & (campaigns["end_date"] <= data_end)
)

print("TRANSACTION OBSERVATION WINDOW")
print("Start:", data_start)
print("End:", data_end)

print("\nCAMPAIGN OBSERVABILITY")
print(
    campaigns[
        ["campaign_id", "start_date", "end_date", "fully_observed"]
    ].sort_values("start_date").to_string(index=False)
)

print("\nSUMMARY")
print("Total campaigns:", len(campaigns))
print("Fully within transaction window:", int(campaigns["fully_observed"].sum()))
print("Not fully within transaction window:", int((~campaigns["fully_observed"]).sum()))
