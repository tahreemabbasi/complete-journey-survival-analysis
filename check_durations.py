
from pathlib import Path
import pandas as pd

cohort = pd.read_csv(
    Path("data") / "survival_cohort.csv",
    parse_dates=["start_date", "end_date", "redemption_date", "observed_end"]
)

cohort["campaign_length_days"] = (
    cohort["end_date"] - cohort["start_date"]
).dt.days

print("CAMPAIGN LENGTHS (days)")
print(cohort.groupby("campaign_id")["campaign_length_days"].first().describe())

print("\nLONGEST OBSERVED DURATIONS")
cols = [
    "household_id", "campaign_id", "start_date", "end_date",
    "redemption_date", "event", "duration_days", "campaign_length_days"
]
print(
    cohort.nlargest(10, "duration_days")[cols].to_string(index=False)
)

print("\nZERO-DAY EVENTS")
zero = cohort[(cohort["event"] == 1) & (cohort["duration_days"] == 0)]
print("Count:", len(zero))
print(zero[cols].head(10).to_string(index=False))

print("\nEVENTS OCCURRING ON CAMPAIGN START DATE")
same_day = cohort[
    (cohort["event"] == 1)
    & (cohort["redemption_date"] == cohort["start_date"])
]
print("Count:", len(same_day))

print("\nDURATION BEYOND CAMPAIGN LENGTH")
beyond = cohort[cohort["duration_days"] > cohort["campaign_length_days"]]
print("Count:", len(beyond))
