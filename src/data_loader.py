import os
import pandas as pd
from nfl_data_py import import_schedules, import_team_desc, importz_team_stats

CACHE_DIR = 'data/raw'
os.makedirs(CACHE_DIR, exist_ok=True)

def load_schedules(seasons, use_cache=True):

    cache_file = f"{CACHE_DIR}/schedules_{min(seasons)}_{max(seasons)}.cvs"

    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached schedules from {cache_file}")
        return pd.read_csv(cache_file)
    
    print(f"Downloading schedules for seasons: {seasons}")
    df = import_schedules(seasons)


    df = df.rename(coloums={
        "team_home": "home_team",
        "team_away": "away_team",
        "game_id": "game_id",
        "game_type": "game_type",
        "week": "week",
        "season": "season",
        "home_score": "home_score",
        "away_score": "away_score",
        "neutral_site": "neutral_site"
    })

    df['playoff'] = df['game_type'].isin([ 'WC', 'DIV', 'CONF', 'SB']).astype(int)

    if use_cache:
        df.to_csv(cache_file, index=False)
        print(f"Cached schedules to {cache_file}")
    return df

def load_team_stats(seasons, use_cache=True):
    """
    Load team statistics (per-season averages).
    Covers passing, rushing, scoring, turnovers, etc.

    Returns: DataFrame with stats per team per season.
    """

    cache_file = f"{CACHE_DIR}/team_stats_{min(seasons)}_{max(seasons)}.csv"

    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached team stats → {cache_file}")
        return pd.read_csv(cache_file)

    print(f"Downloading team stats for seasons {seasons} ...")

    # Pull team stats per season
    stats = import_team_stats(seasons, "REG")  # regular season averages

    # Standardize and keep the columns you need
    keep_cols = [
        "team", "season",
        "passing_yards_pg", "rushing_yards_pg", "total_yards_pg",
        "points_pg", "passing_touchdowns_pg",
        "fumbles_lost_pg", "interceptions_thrown_pg"
    ]

    stats = stats[keep_cols].copy()

    if use_cache:
        stats.to_csv(cache_file, index=False)
        print(f"Saved team stats cache → {cache_file}")

    return stats


def load_data(seasons, use_cache=True):
    """
    Load schedules + team stats together.
    Returns:
        - schedules_df
        - team_stats_df
    """

    schedules = load_schedules(seasons, use_cache=use_cache)
    team_stats = load_team_stats(seasons, use_cache=use_cache)

    # Optional: verify both datasets have the same season range
    missing_seasons = set(seasons) - set(team_stats["season"].unique())
    if missing_seasons:
        print("Warning: Missing team stats for seasons:", missing_seasons)

    return schedules, team_stats