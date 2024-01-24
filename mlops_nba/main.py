import pandas as pd
from pathlib import Path
from mlops_nba.common.io import create_folder
from mlops_nba.common.dates import get_now
from mlops_nba.potential_stars.extract import create_nba_features, stars_definition
from mlops_nba.config import RAW_DATA_DIR, CURATED_DATA_DIR,OUTPUT_DIR
import re
import sys
import logging
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import datetime
from pathlib import Path

from sklearn.base import BaseEstimator


def save_model_params(model, filename):
    """Save model parameters to a file in a simple format."""
    with open(filename, 'w') as file:
        for param, value in model.get_params().items():
            file.write(f"{param}: {value}\n")


# Create a logs folder if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging with a timestamp in the file name
current_time = time.strftime("%Y%m%d-%H%M%S")
log_filename = f'nba_data_processing_{current_time}.log'
logging.basicConfig(filename=LOGS_DIR / log_filename, level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def read_kaggle_data(file_path, encoding='utf-8'):
    """Read the Kaggle dataset with automatic delimiter detection."""
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            first_line = file.readline()
        delimiter = ';' if ';' in first_line else ','
        return pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin1', delimiter=delimiter)

def store_curated_data(players, file_name):
    """Store the processed data in the curated directory."""
    create_folder(CURATED_DATA_DIR)
    current_date = get_now(for_files=True)
    output_file = CURATED_DATA_DIR / f"{file_name}-curated-{current_date}.parquet"
    players.to_parquet(output_file, compression='snappy')

def process_file(file_path):
    """Process a single file and return processed data."""
    start_time = time.time()
    logging.info(f"Processing {file_path.name}...")
    kaggle_data = read_kaggle_data(file_path)

    year_match = re.search(r'\d{4}-\d{4}', file_path.name)
    year = year_match.group() if year_match else 'Unknown'
    kaggle_data['Year'] = year

    processed_data = create_nba_features(kaggle_data)
    processed_data["rising_stars"] = processed_data.apply(stars_definition, axis=1)

    processing_time = time.time() - start_time
    logging.info(f"Data processed for {file_path.name}, DataFrame shape: {processed_data.shape} in {processing_time:.2f} seconds")
    return processed_data

def merge_and_store_data(all_data):
    """Merge and store all processed data in a single Parquet file and return the merged DataFrame."""
    merged_data = pd.concat(all_data, ignore_index=True)
    merged_data.sort_values(by=['Year'], inplace=True)
    create_folder(CURATED_DATA_DIR)
    current_date = get_now(for_files=True)
    output_file = CURATED_DATA_DIR / f"curated_player-{current_date}.parquet"
    merged_data.to_parquet(output_file, compression='snappy')
    logging.info(f"Merged data stored in {output_file}")

    return merged_data  # Returning the merged DataFrame


def train_model(data):
    """Train the model on the given data."""
    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Pos', 'Tm'])
        ])

    # Define model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    # Split data into training and test sets for PTS
    X = data.drop(['Player', 'PTS', 'FG%'], axis=1)
    y_pts = data['PTS']
    X_train_pts, X_test_pts, y_train_pts, y_test_pts = train_test_split(X, y_pts, test_size=0.2, random_state=42)

    # Train model to predict PTS
    model.fit(X_train_pts, y_train_pts)
    pts_preds = model.predict(X_test_pts)

    rmse_pts = mean_squared_error(y_test_pts, pts_preds, squared=False)
    logging.info(f'RMSE for PTS prediction: {rmse_pts}')

    # Generate a timestamp
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Ensure the model output directory exists
    model_output_dir = Path("models/train")
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Save the model with the timestamp in the filename
    model_filename = f'pts_prediction_model_{current_time}.joblib'
    joblib.dump(model, model_output_dir / model_filename)
    logging.info(f"Model saved as {model_filename}")

    # Save model's parameters to a simple format file
    params_filename = f'model_params_{current_time}.txt'
    save_model_params(model, model_output_dir / params_filename)
    logging.info(f"Model parameters saved as {params_filename}")

    return model, X_test_pts, y_test_pts,rmse_pts

def predict_and_store_output(model, X_test, y_test):
    """Make predictions with the model and store the results."""
    predictions = model.predict(X_test)
    output_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    
    # Generate a timestamp
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = OUTPUT_DIR / f"predictions_{current_time}.parquet"

    # Ensure the output directory exists
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_df.to_parquet(output_file, compression='snappy')
    logging.info(f"Predictions stored in {output_file}")
    
if __name__ == "__main__":
    print("Starting NBA data preprocessing...")

    if not create_folder(RAW_DATA_DIR):
        print(f"Raw data directory {RAW_DATA_DIR} exists.")
    else:
        print(f"Raw data directory {RAW_DATA_DIR} created.")

    if not create_folder(CURATED_DATA_DIR):
        print(f"Curated data directory {CURATED_DATA_DIR} exists.")
    else:
        print(f"Curated data directory {CURATED_DATA_DIR} created.")

    raw_data_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not raw_data_files:
        print(f"No CSV files found in {RAW_DATA_DIR}. Please check your data source.")
        sys.exit(1)

    all_processed_data = []
    for file_path in sorted(RAW_DATA_DIR.glob('*.csv'), key=lambda x: x.stat().st_mtime):
        print(f"Processing file: {file_path.name}")
        processed_data = process_file(file_path)
        if processed_data is not None:
            all_processed_data.append(processed_data)
        else:
            print(f"Warning: No data returned for {file_path.name}")

    if all_processed_data:
        merged_data = merge_and_store_data(all_processed_data)
        logging.info("Starting model training...")
        model, X_test, y_test ,rmse_pts= train_model(merged_data)
        logging.info(f"Model training completed with RMSE: {rmse_pts}")

        logging.info("Starting prediction...")
        predict_and_store_output(model, X_test, y_test)
        logging.info("Prediction completed.")
    else:
        logging.info("No data to train model.")

    print("NBA data preprocessing, model training, and prediction completed.")