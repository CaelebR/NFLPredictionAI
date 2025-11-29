import os
import pandas as pd
from nfl_data_py import import_schedules, import_team_desc, import_team_stats

CACHE_DIR = 'data/raw'
os.makedirs(CACHE_DIR, exist_ok=True)

def load_schedules(seasons, use_cache=True):
    """
    Load NFL game schedules for specified seasons.
    
    Args:
        seasons: List of season years (e.g., [2020, 2021, 2022])
        use_cache: Whether to use cached data if available
    
    Returns:
        DataFrame with game schedules
    """
    cache_file = f"{CACHE_DIR}/schedules_{min(seasons)}_{max(seasons)}.csv"

    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached schedules from {cache_file}")
        return pd.read_csv(cache_file)
    
    print(f"Downloading schedules for seasons: {seasons}")
    df = import_schedules(seasons)

    # Check what columns actually exist before renaming
    print(f"Available columns: {df.columns.tolist()}")

    # The nfl_data_py library already uses these column names, so renaming may not be needed
    # Only rename if the columns exist with different names
    rename_map = {}
    if 'team_home' in df.columns:
        rename_map['team_home'] = 'home_team'
    if 'team_away' in df.columns:
        rename_map['team_away'] = 'away_team'
    
    if rename_map:
        df = df.rename(columns=rename_map)