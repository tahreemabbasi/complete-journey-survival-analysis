
from pathlib import Path
import pandas as pd

path = Path("data") / "promotions.csv"

# Read only five rows; do not load the entire 582 MB file.
sample = pd.read_csv(path, nrows=5)

print("PROMOTIONS COLUMNS")
print(sample.columns.tolist())

print("\nFIRST FIVE ROWS")
print(sample.to_string(index=False))

print("\nDATA TYPES")
print(sample.dtypes)
