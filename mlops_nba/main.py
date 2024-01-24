# import runpy
# from pathlib import Path
# import sys

# # Importing necessary modules from your project
# from mlops_nba.common.io import create_folder
# from mlops_nba.config import RAW_DATA_DIR, CURATED_DATA_DIR

# def main():
#     # Ensure the RAW_DATA_DIR and CURATED_DATA_DIR exist
#     if not create_folder(RAW_DATA_DIR):
#         print(f"Raw data directory {RAW_DATA_DIR} exists.")
#     else:
#         print(f"Raw data directory {RAW_DATA_DIR} created.")

#     if not create_folder(CURATED_DATA_DIR):
#         print(f"Curated data directory {CURATED_DATA_DIR} exists.")
#     else:
#         print(f"Curated data directory {CURATED_DATA_DIR} created.")

#     # Check for raw data files
#     raw_data_files = list(RAW_DATA_DIR.glob("*.csv"))
#     if not raw_data_files:
#         print(f"No CSV files found in {RAW_DATA_DIR}. Please check your data source.")
#         sys.exit(1)

#     # Run the extract.py script
#     print("Running extract.py to process raw NBA player stats data...")
#     runpy.run_path(Path("mlops_nba/potential_stars/extract.py").resolve())

# if __name__ == "__main__":
#     main()

# todo batch the kaggle csv file to fake new data then afterwards run analytics (extract.py) on the batch

import pandas as pd
from pathlib import Path

from mlops_nba.common.io import create_folder
from mlops_nba.common.dates import get_now
from mlops_nba.potential_stars.extract import get_raw_data, create_nba_features, stars_definition
from mlops_nba.config import RAW_DATA_DIR, CURATED_DATA_DIR


def read_kaggle_data(file_path, encoding='utf-8'):
    """Read the Kaggle dataset with specified encoding."""
    try:
        return pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        # If utf-8 encoding fails, try a different encoding
        return pd.read_csv(file_path, encoding='latin1')


def store_curated_data(players):
    """Store the processed data in the curated directory."""
    create_folder(CURATED_DATA_DIR)
    current_date = get_now(for_files=True)
    output_file = CURATED_DATA_DIR / f"curated_players-{current_date}.parquet"
    players.to_parquet(output_file, compression='snappy')


if __name__ == "__main__":
    # Path to the downloaded Kaggle dataset
    kaggle_file_path = RAW_DATA_DIR / '2023-2024 NBA Player Stats - Regular.csv'
    
    # Read data from Kaggle dataset
    kaggle_data = read_kaggle_data(kaggle_file_path)

    # Process the data using functions from extract.py
    processed_data = create_nba_features(kaggle_data)
    processed_data["rising_stars"] = processed_data.apply(stars_definition, axis=1)

    # Store the processed data
    store_curated_data(processed_data)

    print("Data pipeline execution complete.")
