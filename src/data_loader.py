import os
import pandas as pd
import nflreadpy as nfl

CACHE_DIR = 'data/raw'
os.makedirs(CACHE_DIR, exist_ok=True)

def load_schedules(seasons, use_cache=True):
    """
    Load NFL game schedules for specified seasons using nflreadpy.
    """
    cache_file = f"{CACHE_DIR}/schedules_{min(seasons)}_{max(seasons)}.csv"

    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached schedules from {cache_file}")
        return pd.read_csv(cache_file)
    
    print(f"Downloading schedules for seasons: {seasons}")
    
    # nflreadpy returns Polars DataFrame, convert to pandas
    df = nfl.load_schedules(seasons=seasons).to_pandas()

    # Add playoff indicator
    df['playoff'] = df['game_type'].isin(['WC', 'DIV', 'CON', 'SB']).astype(int)

    # Add result column for completed games
    if 'home_score' in df.columns and 'away_score' in df.columns:
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
        df['game_completed'] = df['home_score'].notna() & df['away_score'].notna()

    if use_cache:
        df.to_csv(cache_file, index=False)
        print(f"Cached schedules to {cache_file}")
    
    return df


def load_team_stats(seasons, use_cache=True):
    """
    Load team statistics using nflreadpy.
    """
    cache_file = f"{CACHE_DIR}/team_stats_{min(seasons)}_{max(seasons)}.csv"

    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached team stats → {cache_file}")
        return pd.read_csv(cache_file)

    print(f"Loading team stats for seasons {seasons} ...")

    # Load player stats and aggregate to team level
    player_stats = nfl.load_player_stats(seasons=seasons).to_pandas()
    
    # Print columns to debug
    print(f"Available columns: {player_stats.columns.tolist()[:20]}...")
    
    # Find team column
    team_col = None
    for possible_name in ['recent_team', 'team', 'team_abbr', 'posteam']:
        if possible_name in player_stats.columns:
            team_col = possible_name
            break
    
    if team_col is None:
        raise ValueError("Could not find team column in player stats. Available columns: " + str(player_stats.columns.tolist()))
    
    print(f"Using team column: {team_col}")
    
    # Columns to aggregate
    agg_dict = {
        'passing_yards': 'sum',
        'rushing_yards': 'sum',
        'receiving_yards': 'sum',
        'passing_tds': 'sum',
        'rushing_tds': 'sum',
        'receiving_tds': 'sum',
        'completions': 'sum',
        'attempts': 'sum',
        'carries': 'sum',
    }

    # Fumble and interception columns to aggregate if they exist
    fumble_cols = ['rushing_fumbles_lost', 'receiving_fumbles_lost', 'sack_fumbles_lost']
    interception_col = 'passing_interceptions'

    # Add fumble columns to agg_dict if present
    for col in fumble_cols:
        if col in player_stats.columns:
            agg_dict[col] = 'sum'
    # Add interception column if present
    if interception_col in player_stats.columns:
        agg_dict[interception_col] = 'sum'

    # Filter agg_dict for existing columns only
    agg_dict = {k: v for k, v in agg_dict.items() if k in player_stats.columns}

    print(f"Aggregating columns: {list(agg_dict.keys())}")
    
    # Aggregate
    team_stats = player_stats.groupby([team_col, 'season']).agg(agg_dict).reset_index()
    team_stats = team_stats.rename(columns={team_col: 'team'})
    print(team_stats.columns.tolist())

    games_per_season = 17
    
    # Calculate per-game stats
    team_stats['passing_yards_pg'] = team_stats.get('passing_yards', 0) / games_per_season
    team_stats['rushing_yards_pg'] = team_stats.get('rushing_yards', 0) / games_per_season
    team_stats['total_yards_pg'] = (team_stats.get('passing_yards', 0) + team_stats.get('rushing_yards', 0)) / games_per_season
    team_stats['points_pg'] = ((team_stats.get('passing_tds', 0) + team_stats.get('rushing_tds', 0) + team_stats.get('receiving_tds', 0)) * 7) / games_per_season
    team_stats['passing_touchdowns_pg'] = team_stats.get('passing_tds', 0) / games_per_season

    # Sum fumble lost columns safely into one combined column
    team_stats['fumbles_lost'] = 0
    for col in fumble_cols:
        if col in team_stats.columns:
            team_stats['fumbles_lost'] += team_stats[col]
    team_stats['fumbles_lost_pg'] = team_stats['fumbles_lost'] / games_per_season

    # Interceptions per game
    team_stats['interceptions_thrown_pg'] = team_stats.get(interception_col, 0) / games_per_season

    # Drop the raw fumble and interception columns so only summary remain
    cols_to_drop = fumble_cols + [interception_col]
    existing_cols_to_drop = [col for col in cols_to_drop if col in team_stats.columns]
    team_stats = team_stats.drop(columns=existing_cols_to_drop + ['fumbles_lost'])

    if use_cache:
        team_stats.to_csv(cache_file, index=False)
        print(f"Saved team stats cache → {cache_file}")

    return team_stats




def load_data(seasons, use_cache=True):
    """Load schedules + team stats together."""
    schedules = load_schedules(seasons, use_cache=use_cache)
    team_stats = load_team_stats(seasons, use_cache=use_cache)

    missing_seasons = set(seasons) - set(team_stats["season"].unique())
    if missing_seasons:
        print(f"Warning: Missing team stats for seasons: {missing_seasons}")

    print(f"\nLoaded {len(schedules)} games and stats for {len(team_stats)} team-seasons")
    
    return schedules, team_stats