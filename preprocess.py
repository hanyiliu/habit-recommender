# preprocess.py  (project root entry point)
#
# Usage: python preprocess.py
#
# Runs the full data pipeline:
#   raw .dat file -> DataFrame -> category mapping -> 48-slot sequences -> .pkl

from pathlib import Path
from src.data.preprocessing.atus_loader import load_atus_activity_file
from src.utils.activity_map import map_activity_category
from src.data.preprocessing.preprocessor import build_all_sequences, save_sequences

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_PATH    = PROJECT_ROOT / "data" / "2024_data" / "atusact_2024.dat"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences.pkl"

def main():
    print("Step 1: Loading raw ATUS data...")
    df = load_atus_activity_file(RAW_PATH)
    print(f"  Loaded {len(df)} rows, {df['TUCASEID'].nunique()} respondents")

    print("Step 2: Mapping activity categories...")
    df["CATEGORY"] = df.apply(map_activity_category, axis=1)
    print(df["CATEGORY"].value_counts())

    print("Step 3: Building 48-slot sequences...")
    sequences = build_all_sequences(df)

    print("Step 4: Saving sequences...")
    save_sequences(sequences, OUTPUT_PATH)
    print("Done.")

if __name__ == "__main__":
    main()