import pandas as pd
from pathlib import Path
from mlops_nba.common.io import create_folder
from mlops_nba.common.dates import get_now
from mlops_nba.potential_stars.extract import create_nba_features, stars_definition
from mlops_nba.config import RAW_DATA_DIR, CURATED_DATA_DIR
import re

def read_kaggle_data(file_path, encoding='utf-8'):
    """Read the Kaggle dataset with automatic delimiter detection."""
    try:
        # Open the file and read the first line to determine the delimiter
        with open(file_path, 'r', encoding=encoding) as file:
            first_line = file.readline()
        if ';' in first_line:
            delimiter = ';'
        else:
            delimiter = ','

        return pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
    except UnicodeDecodeError:
        # If utf-8 encoding fails, try a different encoding
        return pd.read_csv(file_path, encoding='latin1', delimiter=delimiter)


def store_curated_data(players, file_name):
    """Store the processed data in the curated directory."""
    create_folder(CURATED_DATA_DIR)
    current_date = get_now(for_files=True)
    output_file = CURATED_DATA_DIR / f"{file_name}-curated-{current_date}.parquet"
    players.to_parquet(output_file, compression='snappy')









def process_file(file_path):
    """Process a single file and return processed data."""
    print(f"Processing {file_path.name}...")
    kaggle_data = read_kaggle_data(file_path)

    # Extract the year from the filename (assuming it's always in 'YYYY-YYYY' format)
    year_match = re.search(r'\d{4}-\d{4}', file_path.name)
    if year_match:
        year = year_match.group()
    else:
        year = 'Unknown'

    # Add the year as a new column
    kaggle_data['Year'] = year

    processed_data = create_nba_features(kaggle_data)
    processed_data["rising_stars"] = processed_data.apply(stars_definition, axis=1)
    print(f"Data processed for {file_path.name}, DataFrame shape: {processed_data.shape}")
    return processed_data

def merge_and_store_data(all_data):
    """Merge and store all processed data in a single Parquet file."""
    merged_data = pd.concat(all_data, ignore_index=True)
    
    # Sort by the 'Year' column
    merged_data.sort_values(by=['Year'], inplace=True)

    # Ensure the CURATED_DATA_DIR exists
    create_folder(CURATED_DATA_DIR)

    # Define the output file path with the current date
    current_date = get_now(for_files=True)
    output_file = CURATED_DATA_DIR / f"curated_player-{current_date}.parquet"

    # Save the merged data as a Parquet file
    merged_data.to_parquet(output_file, compression='snappy')

    print(f"Merged data stored in {output_file}")


if __name__ == "__main__":
    all_processed_data = []
    
    for file_path in sorted(RAW_DATA_DIR.glob('*.csv'), key=lambda x: x.stat().st_mtime):
        print("___________________")
        print("START PREPROCESSING")
        print(file_path)
        processed_data = process_file(file_path)
        if processed_data is not None:
            all_processed_data.append(processed_data)
        else:
            print(f"Warning: No data returned for {file_path.name}")

    if all_processed_data:
        merge_and_store_data(all_processed_data)
        print("Preprocessing execution for all files complete.")
    else:
        print("No data to merge.")


        