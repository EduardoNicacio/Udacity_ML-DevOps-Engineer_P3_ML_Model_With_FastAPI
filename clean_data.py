"""Clean the raw Census data by stripping whitespace from columns and values.

The raw ``data/census.csv`` contains leading/trailing spaces in every column
name and in every string value.  This script writes a cleaned copy to
``data/census_clean.csv`` without touching the original file.

Running the script is idempotent: cleaning an already-clean file produces the
same output.
"""

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_PATH = os.path.join(DATA_DIR, "census.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "census_clean.csv")


def clean_data(raw_path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw data and strip whitespace from column names and values.

    Parameters
    ----------
    raw_path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Dataframe with stripped column names and stripped string values.
    """
    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.strip()
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()
    return df


def main() -> None:
    """Clean the raw data and save it to ``data/census_clean.csv``."""
    df = clean_data()
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Wrote {len(df)} rows and {len(df.columns)} columns to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
