
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sksurv.util import Surv
from sksurv.nonparametric import CensoringDistributionEstimator

PROJECT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PROJECT_DIR / "data" / "model_dataset.csv")

df["duration_days"] = pd.to_numeric(
    df["duration_days"], errors="coerce"
)
df["event"] = pd.to_numeric(df["event"], errors="coerce")

df = df.dropna(
    subset=["duration_days", "event", "household_id"]
).copy()
df = df[df["event"].isin([0, 1])]
df = df[df["duration_days"] > 0].copy()
df["event"] = df["event"].astype(int)

groups = df["household_id"].astype(str).to_numpy()

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.25, random_state=42
)
train_idx, test_idx = next(splitter.split(df, groups=groups))

train = df.iloc[train_idx]
test = df.iloc[test_idx]

y_train = Surv.from_arrays(
    event=train["event"].astype(bool).to_numpy(),
    time=train["duration_days"].to_numpy(dtype=float),
)
y_test = Surv.from_arrays(
    event=test["event"].astype(bool).to_numpy(),
    time=test["duration_days"].to_numpy(dtype=float),
)

print("Structured training dtype:", y_train.dtype)
print("Structured test dtype:", y_test.dtype)
print("Training event values:", np.unique(y_train["event"], return_counts=True))
print("Test event values:", np.unique(y_test["event"], return_counts=True))

censoring = CensoringDistributionEstimator().fit(y_train)

print("\nHORIZON CASES AND CONTROLS")
for t in [7, 14, 30]:
    cases = int(((y_test["time"] <= t) & y_test["event"]).sum())
    controls = int((y_test["time"] > t).sum())
    censored_before_t = int(
        ((y_test["time"] <= t) & (~y_test["event"])).sum()
    )

    g_t = censoring.predict_proba(np.asarray([t], dtype=float))[0]

    print(
        f"Day {t}: cases={cases}, controls={controls}, "
        f"censored before/by horizon={censored_before_t}, G(t)={g_t:.6f}"
    )

print("\nUnique training durations around early follow-up:")
print(
    train["duration_days"].value_counts().sort_index().head(20)
)
