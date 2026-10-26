#This commented code was previously used to find out the columns and first 3 rows of each file in the intita lsteps
# from pathlib import Path
# import pandas as pd

# DATA_DIR = Path("data")

# tables = [
#     "campaign_descriptions",
#     "campaigns",
#     "coupon_redemptions",
#     "coupons",
#     "demographics",
#     "products",
# ]

# for name in tables:
#     path = DATA_DIR / f"{name}.csv"
#     df = pd.read_csv(path, nrows=5)
#     print(f"\n--- {name} ---")
#     print("Columns:", df.columns.tolist())
#     print(df.head(3).to_string(index=False))
''' The code below preserves your original table inspection 
and adds data types, missing values, key uniqueness, 
and cohort-to-table key checks needed before analytical dataset construction.'''


from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

# Tables to inspect
tables = [
    "campaign_descriptions",
    "campaigns",
    "coupon_redemptions",
    "coupons",
    "demographics",
    "products",
]

# 1. Inspect columns and sample records
for name in tables:
    path = DATA_DIR / f"{name}.csv"

    if not path.exists():
        print(f"\n--- {name} ---")
        print(f"File not found: {path}")
        continue

    df = pd.read_csv(path, nrows=5)

    print(f"\n--- {name} ---")
    print("Columns:", df.columns.tolist())
    print("Data types:")
    print(df.dtypes.to_string())
    print("Sample records:")
    print(df.head(3).to_string(index=False))


# 2. Inspect key uniqueness and missing values
# Read only the columns needed for these checks.
checks = {
    "campaign_descriptions": {
        "columns": ["campaign_id"],
        "keys": ["campaign_id"],
    },
    "campaigns": {
        "columns": ["household_id", "campaign_id"],
        "keys": ["household_id", "campaign_id"],
    },
    "coupon_redemptions": {
        "columns": ["household_id", "campaign_id", "coupon_upc"],
        "keys": None,  # Multiple redemptions may be valid
    },
    "coupons": {
        "columns": ["coupon_upc", "campaign_id", "product_id"],
        "keys": None,  # One coupon may have multiple product mappings
    },
    "demographics": {
        "columns": ["household_id"],
        "keys": ["household_id"],
    },
    "products": {
        "columns": ["product_id"],
        "keys": ["product_id"],
    },
}

for name, config in checks.items():
    path = DATA_DIR / f"{name}.csv"

    if not path.exists():
        continue

    df = pd.read_csv(path, usecols=config["columns"])

    print(f"\n--- DATA QUALITY: {name} ---")
    print("Rows:", len(df))
    print("Missing values:")
    print(df.isna().sum().to_string())

    keys = config["keys"]
    if keys:
        print(
            f"Duplicate rows by {keys}:",
            df.duplicated(subset=keys).sum()
        )


# 3. Check survival cohort keys against source tables
cohort_path = DATA_DIR / "survival_cohort.csv"

if cohort_path.exists():
    cohort = pd.read_csv(
        cohort_path,
        usecols=["household_id", "campaign_id"]
    )

    print("\n--- SURVIVAL COHORT KEY CHECKS ---")
    print("Cohort rows:", len(cohort))
    print(
        "Duplicate household-campaign pairs:",
        cohort.duplicated(
            subset=["household_id", "campaign_id"]
        ).sum()
    )

    assignments_path = DATA_DIR / "campaigns.csv"
    descriptions_path = DATA_DIR / "campaign_descriptions.csv"
    demographics_path = DATA_DIR / "demographics.csv"

    if assignments_path.exists():
        assignments = pd.read_csv(
            assignments_path,
            usecols=["household_id", "campaign_id"]
        )
        cohort_pairs = cohort.drop_duplicates()
        assignment_pairs = assignments.drop_duplicates()

        check = cohort_pairs.merge(
            assignment_pairs,
            on=["household_id", "campaign_id"],
            how="left",
            indicator=True,
            validate="one_to_one"
        )

        print(
            "Cohort pairs missing from campaign assignments:",
            (check["_merge"] == "left_only").sum()
        )

    if descriptions_path.exists():
        descriptions = pd.read_csv(
            descriptions_path,
            usecols=["campaign_id"]
        )
        missing_campaigns = set(cohort["campaign_id"].dropna()) - set(
            descriptions["campaign_id"].dropna()
        )
        print(
            "Cohort campaign IDs without descriptions:",
            sorted(missing_campaigns)
        )

    if demographics_path.exists():
        households = pd.read_csv(
            demographics_path,
            usecols=["household_id"]
        )
        missing_households = set(cohort["household_id"].dropna()) - set(
            households["household_id"].dropna()
        )
        print(
            "Cohort household IDs without demographics:",
            len(missing_households)
        )

print("\nInspection completed.")


# 4. Inspect transaction dates and shopping-trip identifiers

transactions_path = DATA_DIR / "transactions.csv"

if transactions_path.exists():
    transactions = pd.read_csv(
        transactions_path,
        usecols=[
            "household_id",
            "basket_id",
            "sales_value",
            "transaction_timestamp",
        ],
        parse_dates=["transaction_timestamp"],
    )

    print("\n--- TRANSACTION DATA CHECKS ---")
    print("Rows:", len(transactions))
    print("Columns:", transactions.columns.tolist())

    print(
        "Date range:",
        transactions["transaction_timestamp"].min(),
        "to",
        transactions["transaction_timestamp"].max(),
    )

    print(
        "Missing values:",
        transactions.isna().sum().to_dict(),
    )

    print(
        "Unique households:",
        transactions["household_id"].nunique(),
    )

    print(
        "Unique baskets:",
        transactions["basket_id"].nunique(),
    )

    # Check whether a basket belongs to only one household
    households_per_basket = (
        transactions.groupby("basket_id")["household_id"].nunique()
    )

    print(
        "Baskets linked to multiple households:",
        (households_per_basket > 1).sum(),
    )

    # Check whether each household-basket combination is unique
    print(
        "Duplicate household-basket rows:",
        transactions.duplicated(
            subset=["household_id", "basket_id"]
        ).sum(),
    )

    print("\nSample transactions:")
    print(transactions.head(5).to_string(index=False))

    del transactions
