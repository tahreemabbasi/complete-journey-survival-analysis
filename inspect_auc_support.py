
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sksurv.util import Surv
from sksurv.nonparametric import CensoringDistributionEstimator

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "model_dataset.csv"

df = pd.read_csv(DATA_FILE)

df["duration_days"] = pd.to_numeric(
    df["duration_days"], errors="coerce"
)
df["event"] = pd.to_numeric(df["event"], errors="coerce")

df = df.dropna(
    subset=["duration_days", "event", "household_id"]
).copy()

df = df[df["event"].isin([0, 1])].copy()
df = df[df["duration_days"] > 0].copy()
df["event"] = df["event"].astype(int)

groups = df["household_id"].astype(str).to_numpy()

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(df, groups=groups)
)

train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()

y_train = Surv.from_arrays(
    event=train["event"].astype(bool).to_numpy(),
    time=train["duration_days"].to_numpy(dtype=float),
)

y_test = Surv.from_arrays(
    event=test["event"].astype(bool).to_numpy(),
    time=test["duration_days"].to_numpy(dtype=float),
)

censoring = CensoringDistributionEstimator().fit(y_train)

print("TRAINING FOLLOW-UP")
print("Minimum:", y_train["time"].min())
print("Maximum:", y_train["time"].max())
print("Events:", y_train["event"].sum())
print("Censored:", (~y_train["event"]).sum())

print("\nTEST FOLLOW-UP")
print("Minimum:", y_test["time"].min())
print("Maximum:", y_test["time"].max())
print("Events:", y_test["event"].sum())
print("Censored:", (~y_test["event"]).sum())

print("\nCENSORING SURVIVAL ESTIMATES")
for t in [7, 14, 30, 60, 64, 65]:
    try:
        value = censoring.predict_proba(
            np.asarray([t], dtype=float)
        )[0]
        print(f"Day {t}: G(t) = {value:.8f}")
    except Exception as exc:
        print(f"Day {t}: ERROR: {type(exc).__name__}: {exc}")

print("\nTEST OBSERVATIONS AT DAY 65")
print(
    test.loc[
        test["duration_days"] == 65,
        ["duration_days", "event", "household_id"],
    ].groupby(["duration_days", "event"]).size()
)

print("\nFollow-up time frequencies near the upper tail:")
print(
    test["duration_days"]
    .value_counts()
    .sort_index()
    .tail(15)
)
