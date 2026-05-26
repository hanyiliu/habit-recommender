from pathlib import Path
from src.data.preprocessing.atus_loader import load_atus_activity_file

PROJECT_ROOT = Path(__file__).resolve().parent

path = PROJECT_ROOT / "data" / "2024_data" / "atusact_2024.dat"

print(path)

df = load_atus_activity_file(path)

print(df.shape)
print(df.head())
print(df.columns.tolist())