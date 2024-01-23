import re
from pathlib import Path

import pandas as pd

from mlops_nba.common.dates import get_now
from mlops_nba.common.io import create_folder
from mlops_nba.config import CURATED_DATA_DIR, RAW_DATA_DIR

OUTPUT_FILENAME = "curated_players"
# can be automated in next versions
AGE_THRESHOLD = 23
POINTS_THRESHOLD = 10
EFFICENCY_THRESHOLD = 12


def get_file(filename: str):
    """Load intermediate data from curated data directory."""
    df = pd.read_csv(filename, sep=";", encoding="Windows-1252")
    name = filename.stem
    season = re.search(r"\d{4}-\d{4}", name).group()
    type_of_matches = name.split("-").pop().strip()

    df["filename"] = name
    df["season"] = season
    df["period"] = type_of_matches
    return df


def get_raw_data(path: Path):
    """Load intermediate data from curated data directory."""
    files = path.glob("*.csv")
    players = pd.concat(
        [get_file(filename=stat_player) for stat_player in files],
        ignore_index=True,
    )
    return players


def create_nba_features(players: pd.DataFrame) -> pd.DataFrame:
    """Create features from raw data."""
    players["efficency"] = (
        players.PTS
        + players.TRB
        + players.AST
        + players.STL
        + players.BLK
        - (players.FGA - players.FG)
        - (players.FTA - players.FT)
        - players.TOV
    )
    return players


def stars_definition(row):
    """Define rising stars."""
    return (
        row["efficency"] >= EFFICENCY_THRESHOLD
        and row["PTS"] >= POINTS_THRESHOLD
        and row["Age"] <= AGE_THRESHOLD
    )


if __name__ == "__main__":
    current_date = get_now(for_files=True)
    _ = create_folder(path=CURATED_DATA_DIR)

    players = get_raw_data(path=RAW_DATA_DIR)
    players = create_nba_features(players=players)
    players["rising_stars"] = players.apply(stars_definition, axis=1)

    players.to_parquet(
        CURATED_DATA_DIR / f"{OUTPUT_FILENAME}-{current_date}.parquet",
        compression="snappy",
    )
