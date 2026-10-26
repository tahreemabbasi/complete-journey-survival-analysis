
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test


# ============================================================
# 1. FILE PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_FILE = PROJECT_DIR / "data" / "model_dataset.csv"

OUTPUT_DIR = PROJECT_DIR / "model_results" / "kaplan_meier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD AND VALIDATE THE DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")
print("Number of rows:", len(df))
print("Available columns:", df.columns.tolist())

required_columns = [
    "duration_days",
    "event",
    "campaign_type",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        "Check the column names in data/model_dataset.csv."
    )

# Keep only the columns needed for this method.
df = df[required_columns].copy()

# Convert the outcome and duration columns to numeric values.
df["duration_days"] = pd.to_numeric(
    df["duration_days"], errors="coerce"
)

df["event"] = pd.to_numeric(
    df["event"], errors="coerce"
)

# Remove rows with missing or invalid survival information.
df = df.dropna(
    subset=["duration_days", "event", "campaign_type"]
).copy()

# Event must be 1 (redemption) or 0 (censored).
df = df[df["event"].isin([0, 1])].copy()

# Negative survival times are invalid.
df = df[df["duration_days"] >= 0].copy()

df["event"] = df["event"].astype(int)

if df.empty:
    raise ValueError(
        "No valid observations remain after data validation."
    )

if df["event"].sum() == 0:
    raise ValueError(
        "No redemption events were found. "
        "Kaplan-Meier analysis cannot estimate event timing "
        "without any observed events."
    )

print("\nData validation complete.")
print("Valid observations:", len(df))
print("Observed redemptions:", int(df["event"].sum()))
print("Censored observations:", int((df["event"] == 0).sum()))
print(
    "Zero-day observations:",
    int((df["duration_days"] == 0).sum())
)

print("\nObservations by campaign type:")
print(df["campaign_type"].value_counts().sort_index())


# ============================================================
# 3. OVERALL KAPLAN-MEIER CURVE
# ============================================================

kmf = KaplanMeierFitter()

kmf.fit(
    durations=df["duration_days"],
    event_observed=df["event"],
    label="All campaigns"
)

# Save the estimated survival probabilities over time.
overall_survival = kmf.survival_function_.reset_index()
overall_survival.columns = [
    "duration_days",
    "probability_not_yet_redeemed"
]

overall_survival.to_csv(
    OUTPUT_DIR / "overall_survival_estimates.csv",
    index=False
)

# Plot the overall survival curve.
plt.figure(figsize=(9, 6))

kmf.plot_survival_function(
    ci_show=True,
    show_censors=True
)

plt.title("Kaplan-Meier Curve: All Campaigns")
plt.xlabel("Days since campaign started")
plt.ylabel("Probability of not yet redeeming")
plt.ylim(0, 1.02)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "overall_kaplan_meier.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nOverall Kaplan-Meier curve saved.")


# ============================================================
# 4. COMPARE SURVIVAL CURVES BY CAMPAIGN TYPE
# ============================================================

plt.figure(figsize=(9, 6))

for campaign in sorted(df["campaign_type"].unique()):
    group = df[df["campaign_type"] == campaign]

    group_kmf = KaplanMeierFitter()

    group_kmf.fit(
        durations=group["duration_days"],
        event_observed=group["event"],
        label=f"Campaign {campaign} (n={len(group)})"
    )

    group_kmf.plot_survival_function(
        ci_show=False,
        show_censors=False
    )

plt.title("Kaplan-Meier Curves by Campaign Type")
plt.xlabel("Days since campaign started")
plt.ylabel("Probability of not yet redeeming")
plt.ylim(0, 1.02)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "survival_by_campaign_type.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Campaign-type comparison plot saved.")


# ============================================================
# 5. SUMMARY STATISTICS BY CAMPAIGN TYPE
# ============================================================

summary_rows = []

for campaign in sorted(df["campaign_type"].unique()):
    group = df[df["campaign_type"] == campaign]

    group_kmf = KaplanMeierFitter()

    group_kmf.fit(
        durations=group["duration_days"],
        event_observed=group["event"]
    )

    summary_rows.append({
        "campaign_type": campaign,
        "observations": len(group),
        "redemptions": int(group["event"].sum()),
        "censored": int((group["event"] == 0).sum()),
        "observed_redemption_rate": group["event"].mean(),
        "median_survival_days": group_kmf.median_survival_time_,
        "survival_probability_at_day_7": float(
            group_kmf.predict(7)
        ),
        "survival_probability_at_day_14": float(
            group_kmf.predict(14)
        ),
        "survival_probability_at_day_30": float(
            group_kmf.predict(30)
        )
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_DIR / "campaign_type_summary.csv",
    index=False
)

print("\nSummary by campaign type:")
print(summary.to_string(index=False))


# ============================================================
# 6. LOG-RANK TEST
# ============================================================

# Tests whether the survival curves differ across campaign types.
number_of_campaign_types = df["campaign_type"].nunique()

logrank_file = OUTPUT_DIR / "logrank_test.txt"

with open(logrank_file, "w", encoding="utf-8") as file:

    if number_of_campaign_types < 2:
        message = (
            "Log-rank test not performed: "
            "fewer than two campaign types are present."
        )

        print("\n" + message)
        file.write(message)

    else:
        result = multivariate_logrank_test(
            event_durations=df["duration_days"],
            groups=df["campaign_type"],
            event_observed=df["event"]
        )

        output = (
            "Log-rank test comparing campaign types\n"
            "=======================================\n"
            f"Test statistic: {result.test_statistic:.4f}\n"
            f"P-value: {result.p_value:.6g}\n\n"
            "Interpretation:\n"
        )

        if result.p_value < 0.05:
            output += (
                "There is statistical evidence that at least "
                "one campaign type has a different survival curve. "
                "This test does not identify which specific pairs "
                "differ.\n"
            )
        else:
            output += (
                "There is insufficient statistical evidence to "
                "conclude that the survival curves differ across "
                "campaign types.\n"
            )

        print("\n" + output)
        file.write(output)


# ============================================================
# 7. FINISH
# ============================================================

print("\nKaplan-Meier analysis completed.")
print("Results saved to:")
print(OUTPUT_DIR)
