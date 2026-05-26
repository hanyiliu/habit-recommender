import pandas as pd
import numpy as np
from pathlib import Path

# this file will parse raw ATUS data files into DataFrames
#PROJECT_ROOT = Path(__file__).resolve().parents[3]

#raw_data_path = PROJECT_ROOT / "data" / "2024_data" / "atusact_2024.dat"
#do_file_path = PROJECT_ROOT / "data" / "2024_data" / "atusact_2024.do"

#print(raw_data_path)
#print(do_file_path)

def load_atus_activity_file(path):
    """
    Load ATUS activity file into a pandas DataFrame.
    """
    path = Path(path)

    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep="\t")

    # Standardize column names
    df.columns = [col.strip().upper() for col in df.columns]

    return df
