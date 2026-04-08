"""
EuroMetrics Data Pipeline
=========================
Fetches Euroleague game data, calculates advanced stats,
and outputs JSON files for the website.

Usage:
    python pipeline.py               # Update current season (2025)
    python pipeline.py --season 2024 # Backfill a specific season
    python pipeline.py --all         # Backfill all available seasons

Requirements:
    pip install euroleague-api pandas
"""

import os
import json
import argparse
import math
from datetime import datetime, timezone

import pandas as pd
from euroleague_api.game_stats import GameStats
from euroleague_api.player_stats import PlayerStats
from euroleague_api.standings import Standings
from euroleague_api.game_metadata import GameMetadata

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

COMPETITION = "E"           # "E" = Euroleague, "U" = EuroCup
CURRENT_SEASON = 2025       # Start year of 2025-26 season
ALL_SEASONS = list(range(2007, 2026))  # Adjust start year as needed
DATA_DIR = "data"


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def safe_div(numerator, denominator, default=0.0):
    """Division that returns default when denominator is 0."""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def round2(val):
    """Round to 2 decimal places, handle NaN."""
    if pd.isna(val) or not math.isfinite(val):
        return 0.0
    return round(float(val), 2)


def round1(val):
    """Round to 1 decimal place, handle NaN."""
    if pd.isna(val) or not math.isfinite(val):
        return 0.0
    return round(float(val), 1)


def season_label(season_int):
    """Convert 2025 -> '2025-26'"""
    return f"{season_int}-{str(season_int + 1)[-2:]}"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ─────────────────────────────────────────
# STEP 1 — FETCH RAW GAME DATA
# ─────────────────────────────────────────

def fetch_season_games(season):
    """
    Fetch all completed game box scores for a season.
    Returns two DataFrames: team_games, player_games
    """
    print(f"\n[{season_label(season)}] Fetching game list...")

    gm = GameMetadata(COMPETITION)
    gs = GameStats(COMPETITION)

    # Get all game codes for this season
    try:
        meta_df = gm.get_gamecodes_season(season)
    except Exception as e:
        print(f"  ERROR fetching game list: {e}")
        return None, None

    if meta_df is None or meta_df.empty:
        print(f"  No games found for season {season}.")
        return None, None

    # Filter to completed games only (score is not null / both teams have points)
    # The metadata df has columns like: Season, Gamecode, Round, Phase, Home, Away, etc.
    # We only process games that have been played
    print(f"  Found {len(meta_df)} games in metadata.")
    print(f"  Metadata columns: {meta_df.columns.tolist()}")

    # Auto-detect the gamecode column name
    gamecode_col = None
    for candidate in ["Gamecode", "gamecode", "GameCode", "game_code", "GAMECODE", "gamenumber", "GameNumber"]:
        if candidate in meta_df.columns:
            gamecode_col = candidate
            break
    if gamecode_col is None:
        print(f"  ERROR: Could not find gamecode column. Columns: {meta_df.columns.tolist()}")
        return None, None
    print(f"  Using gamecode column: '{gamecode_col}'")

    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    round_col  = find_col(meta_df, ["Round", "round", "RoundNumber", "round_number"])
    phase_col  = find_col(meta_df, ["Phase", "phase", "PhaseType", "phase_type"])
    date_col   = find_col(meta_df, ["date", "Date", "GameDate", "game_date"])
    home_col   = find_col(meta_df, ["hometeam", "Home", "home", "HomeTeam", "home_team", "HomeClub"])
    away_col   = find_col(meta_df, ["awayteam", "Away", "away", "AwayTeam", "away_team", "AwayClub"])
    hscore_col = find_col(meta_df, ["homescore", "HomeScore", "home_score"])
    ascore_col = find_col(meta_df, ["awayscore", "AwayScore", "away_score"])
    played_col = find_col(meta_df, ["played", "Played", "IsPlayed"])

    all_team_rows = []
    all_player_rows = []

    for _, game_row in meta_df.iterrows():
        raw_code = str(game_row[gamecode_col])
        # Handle format like 'E2025_1' -> extract the number after underscore
        if '_' in raw_code:
            game_code = int(raw_code.split('_')[-1])
        else:
            game_code = int(raw_code)

        try:
            # get_game_stats returns player-level box scores we aggregate to team level
            game_df = gs.get_game_stats(season, game_code)

            if game_df is None or game_df.empty:
                continue  # Game not yet played

            if len(all_team_rows) == 0:
                print(f"  get_game_stats columns: {game_df.columns.tolist()}")

            # Add game metadata
            game_df["Season"]   = season
            game_df["Gamecode"] = game_code
            game_df["Round"]    = game_row.get(round_col, None) if round_col else None
            game_df["Phase"]    = game_row.get(phase_col, None) if phase_col else None
            game_df["Date"]     = game_row.get(date_col, None)  if date_col else None
            game_df["Home"]     = game_row.get(home_col, None)  if home_col else None
            game_df["Away"]     = game_row.get(away_col, None)  if away_col else None

            all_team_rows.append(game_df)

            # Player box scores skipped here — fetched in bulk below

        except Exception as e:
            # Game likely not played yet — skip silently
            continue

    if not all_team_rows:
        print(f"  No completed games found yet.")
        return None, None

    team_games = pd.concat(all_team_rows, ignore_index=True)

    # ── Fetch player stats in bulk using PlayerStats class ───────────────────
    print(f"  Fetching player stats for season {season}...")
    player_games = pd.DataFrame()
    try:
        from euroleague_api.player_stats import PlayerStats
        ps = PlayerStats(COMPETITION)
        # get_player_stats returns season aggregates — we use this for per-game averages
        # Correct method is get_player_stats with season as a param
        player_games = ps.get_player_stats(
            endpoint="traditional",
            params={"SeasonCode": f"E{season}"},
            statistic_mode="PerGame"
        )
        if player_games is not None and not player_games.empty:
            player_games["Season"] = season
            print(f"  Player columns: {player_games.columns.tolist()}")
        else:
            player_games = pd.DataFrame()
            print(f"  No player data returned.")
    except Exception as e:
        print(f"  Player stats fetch error: {e}")
        player_games = pd.DataFrame()

    print(f"  Fetched {len(team_games)} team-game rows, {len(player_games)} player rows.")
    return team_games, player_games


# ─────────────────────────────────────────
# STEP 2 — FETCH SCHEDULE & RESULTS
# ─────────────────────────────────────────

def fetch_schedule(season):
    """
    Fetch full schedule including results for completed games.
    Returns a list of game dicts.
    """
    gm = GameMetadata(COMPETITION)

    try:
        meta_df = gm.get_gamecodes_season(season)
    except Exception as e:
        print(f"  ERROR fetching schedule: {e}")
        return []

    if meta_df is None or meta_df.empty:
        return []

    def fc(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    gc_col     = fc(meta_df, ["gamecode", "Gamecode", "GameCode", "gameCode"])
    round_col  = fc(meta_df, ["Round", "round"])
    phase_col  = fc(meta_df, ["Phase", "phase"])
    date_col   = fc(meta_df, ["date", "Date"])
    home_col   = fc(meta_df, ["hometeam", "Home", "home", "HomeTeam"])
    away_col   = fc(meta_df, ["awayteam", "Away", "away", "AwayTeam"])
    hscore_col = fc(meta_df, ["homescore", "HomeScore", "home_score"])
    ascore_col = fc(meta_df, ["awayscore", "AwayScore", "away_score"])
    played_col = fc(meta_df, ["played", "Played", "IsPlayed"])

    schedule = []

    for _, row in meta_df.iterrows():
        raw_code = str(row[gc_col]) if gc_col else ""
        if "_" in raw_code:
            game_code = int(raw_code.split("_")[-1])
        else:
            try:
                game_code = int(raw_code)
            except ValueError:
                continue

        played = bool(row[played_col]) if played_col else False
        home_score = None
        away_score = None
        if played:
            try:
                home_score = int(row[hscore_col]) if hscore_col else None
                away_score = int(row[ascore_col]) if ascore_col else None
            except (ValueError, TypeError):
                pass

        game = {
            "gamecode":   game_code,
            "round":      row[round_col]  if round_col  else None,
            "phase":      row[phase_col]  if phase_col  else None,
            "date":       str(row[date_col]) if date_col else None,
            "home":       row[home_col]   if home_col   else None,
            "away":       row[away_col]   if away_col   else None,
            "home_score": home_score,
            "away_score": away_score,
            "played":     played,
        }
        schedule.append(game)

    return schedule

def calculate_team_stats(team_games):
    """
    Aggregate raw game data and calculate all advanced metrics per team.
    get_game_stats returns one row per game with local.total.* and road.total.* columns.
    """

    def g(row, col, default=0.0):
        """Safely get a numeric value from a row."""
        val = row.get(col, default)
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    # ── Build per-team per-game rows ─────────────────────────────────────────
    # Detect which prefix has team stats — local.team.* or local.total.*
    sample_row = team_games.iloc[0] if not team_games.empty else None
    if sample_row is not None:
        has_total = any("local.total." in c for c in team_games.columns)
        has_team  = any("local.team." in c for c in team_games.columns)
        stat_prefix = "total" if has_total else "team"
        print(f"  Using stat prefix: '{stat_prefix}' (has_total={has_total}, has_team={has_team})")
    else:
        stat_prefix = "total"

    games = []
    for _, row in team_games.iterrows():
        gamecode  = row.get("Gamecode", None)
        round_val = row.get("Round", None)
        phase_val = row.get("Phase", None)
        date_val  = row.get("Date", None)
        # Team names come from the metadata columns we added
        home_name = row.get("Home", None)
        away_name = row.get("Away", None)

        for side, opp, team_name in [("local", "road", home_name), ("road", "local", away_name)]:
            if not team_name or str(team_name).strip() == "":
                # Fallback: try club name columns
                team_name = (row.get(f"{side}.club.name") or
                             row.get(f"{side}.club.editorialName") or
                             row.get(f"{side}.club.abbreviatedName") or
                             f"Unknown_{side}")

            def t(stat):
                return g(row, f"{side}.{stat_prefix}.{stat}")

            def o(stat):
                return g(row, f"{opp}.{stat_prefix}.{stat}")

            pts  = t("points")
            fgm2 = t("fieldGoalsMade2")
            fga2 = t("fieldGoalsAttempted2")
            fgm3 = t("fieldGoalsMade3")
            fga3 = t("fieldGoalsAttempted3")
            ftm  = t("freeThrowsMade")
            fta  = t("freeThrowsAttempted")
            oreb = t("offensiveRebounds")
            dreb = t("defensiveRebounds")
            treb = t("totalRebounds")
            ast  = t("assistances")
            stl  = t("steals")
            tov  = t("turnovers")
            blk  = t("blocksFavour")
            blka = t("blocksAgainst")
            pf   = t("foulsCommited")
            fd   = t("foulsReceived")
            fgm  = fgm2 + fgm3
            fga  = fga2 + fga3

            opp_pts  = o("points")
            opp_fgm2 = o("fieldGoalsMade2")
            opp_fga2 = o("fieldGoalsAttempted2")
            opp_fgm3 = o("fieldGoalsMade3")
            opp_fga3 = o("fieldGoalsAttempted3")
            opp_ftm  = o("freeThrowsMade")
            opp_fta  = o("freeThrowsAttempted")
            opp_oreb = o("offensiveRebounds")
            opp_dreb = o("defensiveRebounds")
            opp_treb = o("totalRebounds")
            opp_ast  = o("assistances")
            opp_stl  = o("steals")
            opp_tov  = o("turnovers")
            opp_blk  = o("blocksFavour")
            opp_blka = o("blocksAgainst")
            opp_pf   = o("foulsCommited")
            opp_fd   = o("foulsReceived")
            opp_fgm  = opp_fgm2 + opp_fgm3
            opp_fga  = opp_fga2 + opp_fga3

            games.append({
                "Team": team_name, "Gamecode": gamecode,
                "Round": round_val, "Phase": phase_val, "Date": date_val,
                "PTS": pts, "FGM2": fgm2, "FGA2": fga2,
                "FGM3": fgm3, "FGA3": fga3, "FGM": fgm, "FGA": fga,
                "FTM": ftm, "FTA": fta, "OREB": oreb, "DREB": dreb,
                "TREB": treb, "AST": ast, "STL": stl, "TOV": tov,
                "BLK": blk, "BLKA": blka, "PF": pf, "FD": fd,
                "OPP_PTS": opp_pts, "OPP_FGM2": opp_fgm2, "OPP_FGA2": opp_fga2,
                "OPP_FGM3": opp_fgm3, "OPP_FGA3": opp_fga3,
                "OPP_FGM": opp_fgm, "OPP_FGA": opp_fga,
                "OPP_FTM": opp_ftm, "OPP_FTA": opp_fta,
                "OPP_OREB": opp_oreb, "OPP_DREB": opp_dreb, "OPP_TREB": opp_treb,
                "OPP_AST": opp_ast, "OPP_STL": opp_stl, "OPP_TOV": opp_tov,
                "OPP_BLK": opp_blk, "OPP_BLKA": opp_blka,
                "OPP_PF": opp_pf, "OPP_FD": opp_fd,
            })

    enriched = pd.DataFrame(games)
    enriched = enriched[enriched["Team"].notna() & (enriched["Team"] != "")]

    # ── Aggregate totals per team ─────────────────────────────────────────────
    stat_cols = ["PTS","FGM2","FGA2","FGM3","FGA3","FGM","FGA",
                 "FTM","FTA","OREB","DREB","TREB","AST","STL","TOV",
                 "BLK","BLKA","PF","FD",
                 "OPP_PTS","OPP_FGM2","OPP_FGA2","OPP_FGM3","OPP_FGA3",
                 "OPP_FGM","OPP_FGA","OPP_FTM","OPP_FTA",
                 "OPP_OREB","OPP_DREB","OPP_TREB",
                 "OPP_AST","OPP_STL","OPP_TOV",
                 "OPP_BLK","OPP_BLKA","OPP_PF","OPP_FD"]

    for col in stat_cols:
        enriched[col] = pd.to_numeric(enriched[col], errors="coerce").fillna(0)

    agg_dict = {"Gamecode": "count"}
    for col in stat_cols:
        agg_dict[col] = "sum"

    totals = enriched.groupby("Team").agg(agg_dict).reset_index()
    totals.rename(columns={"Gamecode": "GP"}, inplace=True)

    team_stats = []

    for _, t in totals.iterrows():
        gp = t["GP"]
        if gp == 0:
            continue

        # ── Possessions ───────────────────────────────────────────────────────
        o_poss = t["FGA"] - t["OREB"] + t["TOV"] + (0.475 * t["FTA"])
        d_poss = t["OPP_FGA"] - t["OPP_OREB"] + t["OPP_TOV"] + (0.475 * t["OPP_FTA"])

        # ── Tempo ─────────────────────────────────────────────────────────────
        tempo = safe_div(d_poss, gp)

        # ── Efficiency ────────────────────────────────────────────────────────
        o_eff   = safe_div(t["PTS"] * 100, o_poss)
        d_eff   = safe_div(t["OPP_PTS"] * 100, d_poss)
        net_rtg = o_eff - d_eff

        # ── PIR ───────────────────────────────────────────────────────────────
        missed_fg = t["FGA"] - t["FGM"]
        missed_ft = t["FTA"] - t["FTM"]
        pir_total = ((t["PTS"] + t["TREB"] + t["AST"] + t["STL"] + t["BLK"] + t["FD"])
                     - (missed_fg + missed_ft + t["TOV"] + t["BLKA"] + t["PF"]))
        pir_pg = safe_div(pir_total, gp)

        # ── Four Factors ──────────────────────────────────────────────────────
        o_efg      = safe_div(t["FGM"] + 0.5 * t["FGM3"], t["FGA"])
        o_tov_pct  = safe_div(t["TOV"] * 100, o_poss)
        o_oreb_pct = safe_div(t["OREB"] * 100, t["OREB"] + t["OPP_DREB"])
        o_fta_fga  = safe_div(t["FTA"], t["FGA"])

        d_efg      = safe_div(t["OPP_FGM"] + 0.5 * t["OPP_FGM3"], t["OPP_FGA"])
        d_tov_pct  = safe_div(t["OPP_TOV"] * 100, d_poss)
        d_oreb_pct = safe_div(t["OPP_OREB"] * 100, t["OPP_OREB"] + t["DREB"])
        d_fta_fga  = safe_div(t["OPP_FTA"], t["OPP_FGA"])

        # ── Basics ────────────────────────────────────────────────────────────
        o_3p_pct   = safe_div(t["FGM3"] * 100, t["FGA3"])
        o_2p_pct   = safe_div(t["FGM2"] * 100, t["FGA2"])
        o_ft_pct   = safe_div(t["FTM"] * 100, t["FTA"])
        o_blk_pct  = safe_div(t["BLK"] * 100, t["OPP_FGA"])
        o_stl_pct  = safe_div(t["STL"] * 100, d_poss)
        o_nstov    = safe_div((t["TOV"] - t["OPP_STL"]) * 100, o_poss)

        d_3p_pct   = safe_div(t["OPP_FGM3"] * 100, t["OPP_FGA3"])
        d_2p_pct   = safe_div(t["OPP_FGM2"] * 100, t["OPP_FGA2"])
        d_ft_pct   = safe_div(t["OPP_FTM"] * 100, t["OPP_FTA"])
        d_blk_pct  = safe_div(t["BLKA"] * 100, t["FGA"])
        d_stl_pct  = safe_div(t["OPP_STL"] * 100, o_poss)
        d_nstov    = safe_div((t["OPP_TOV"] - t["STL"]) * 100, d_poss)

        # ── Characteristics ───────────────────────────────────────────────────
        o_3pa_fga  = safe_div(t["FGA3"] * 100, t["FGA"])
        o_ast_fgm  = safe_div(t["AST"] * 100, t["FGM"])
        d_3pa_fga  = safe_div(t["OPP_FGA3"] * 100, t["OPP_FGA"])
        d_ast_fgm  = safe_div(t["OPP_AST"] * 100, t["OPP_FGM"])

        # ── Point Distribution ─────────────────────────────────────────────────
        o_pts_3_pct  = safe_div(t["FGM3"] * 3 * 100, t["PTS"])
        o_pts_2_pct  = safe_div(t["FGM2"] * 2 * 100, t["PTS"])
        o_pts_ft_pct = safe_div(t["FTM"] * 100, t["PTS"])

        d_pts_3_pct  = safe_div(t["OPP_FGM3"] * 3 * 100, t["OPP_PTS"])
        d_pts_2_pct  = safe_div(t["OPP_FGM2"] * 2 * 100, t["OPP_PTS"])
        d_pts_ft_pct = safe_div(t["OPP_FTM"] * 100, t["OPP_PTS"])

        team_stats.append({
            "team":       t["Team"],
            "gp":         int(gp),
            "pir":        round1(pir_pg),
            "o_eff":      round2(o_eff),
            "d_eff":      round2(d_eff),
            "net_rtg":    round2(net_rtg),
            "tempo":      round2(tempo),
            "o_efg":      round2(o_efg * 100),
            "d_efg":      round2(d_efg * 100),
            "o_tov_pct":  round2(o_tov_pct),
            "d_tov_pct":  round2(d_tov_pct),
            "o_oreb_pct": round2(o_oreb_pct),
            "d_oreb_pct": round2(d_oreb_pct),
            "o_fta_fga":  round2(o_fta_fga),
            "d_fta_fga":  round2(d_fta_fga),
            "o_3p_pct":   round2(o_3p_pct),
            "d_3p_pct":   round2(d_3p_pct),
            "o_2p_pct":   round2(o_2p_pct),
            "d_2p_pct":   round2(d_2p_pct),
            "o_ft_pct":   round2(o_ft_pct),
            "d_ft_pct":   round2(d_ft_pct),
            "o_blk_pct":  round2(o_blk_pct),
            "d_blk_pct":  round2(d_blk_pct),
            "o_stl_pct":  round2(o_stl_pct),
            "d_stl_pct":  round2(d_stl_pct),
            "o_nstov_pct":round2(o_nstov),
            "d_nstov_pct":round2(d_nstov),
            "o_3pa_fga":  round2(o_3pa_fga),
            "d_3pa_fga":  round2(d_3pa_fga),
            "o_ast_fgm":  round2(o_ast_fgm),
            "d_ast_fgm":  round2(d_ast_fgm),
            "o_pts_3_pct":  round2(o_pts_3_pct),
            "d_pts_3_pct":  round2(d_pts_3_pct),
            "o_pts_2_pct":  round2(o_pts_2_pct),
            "d_pts_2_pct":  round2(d_pts_2_pct),
            "o_pts_ft_pct": round2(o_pts_ft_pct),
            "d_pts_ft_pct": round2(d_pts_ft_pct),
            "pts_pg":     round1(safe_div(t["PTS"], gp)),
            "opp_pts_pg": round1(safe_div(t["OPP_PTS"], gp)),
            "ast_pg":     round1(safe_div(t["AST"], gp)),
            "treb_pg":    round1(safe_div(t["TREB"], gp)),
            "stl_pg":     round1(safe_div(t["STL"], gp)),
            "blk_pg":     round1(safe_div(t["BLK"], gp)),
            "tov_pg":     round1(safe_div(t["TOV"], gp)),
        })

    return team_stats


def calculate_player_stats(player_games):
    """
    Calculate player stats from season-aggregate PerGame data
    returned by PlayerStats.get_player_stats_season().
    Column names printed on first run so we can verify mapping.
    """
    if player_games is None or (hasattr(player_games, "empty") and player_games.empty):
        return []

    df = player_games.copy()

    # ── Map column names ──────────────────────────────────────────────────────
    # PlayerStats PerGame columns use full names like 'Score', 'FieldGoalsMade2', etc.
    col_map = {
        # Actual column names from get_player_stats PerGame endpoint
        "player.name":              "Name",
        "player.team.name":         "Team",
        "player.team.code":         "TeamCode",
        "player.imageUrl":          "ImageUrl",
        "gamesPlayed":              "GP",
        "minutesPlayed":            "MIN_STR",
        "pointsScored":             "PTS",
        "twoPointersMade":          "FGM2",
        "twoPointersAttempted":     "FGA2",
        "threePointersMade":        "FGM3",
        "threePointersAttempted":   "FGA3",
        "freeThrowsMade":           "FTM",
        "freeThrowsAttempted":      "FTA",
        "offensiveRebounds":        "OREB",
        "defensiveRebounds":        "DREB",
        "totalRebounds":            "TREB",
        "assists":                  "AST",
        "steals":                   "STL",
        "turnovers":                "TOV",
        "blocks":                   "BLK",
        "blocksAgainst":            "BLKA",
        "foulsCommited":            "PF",
        "foulsDrawn":               "FD",
        "pir":                      "PIR_API",
    }

    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure required columns exist
    required = ["Name", "Team", "GP"]
    for col in required:
        if col not in df.columns:
            print(f"  WARNING: Missing column '{col}'. Available: {df.columns.tolist()}")
            return []

    num_cols = ["GP","PTS","FGM2","FGA2","FGM3","FGA3",
                "FTM","FTA","OREB","DREB","TREB","AST","STL",
                "TOV","BLK","BLKA","PF","FD"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0
    # Derive total FGM and FGA from 2s and 3s
    df["FGM"] = df["FGM2"] + df["FGM3"]
    df["FGA"] = df["FGA2"] + df["FGA3"]

    # Parse minutes — PerGame data may give MM:SS or float
    def parse_minutes(val):
        try:
            if pd.isna(val) or val == "" or val == "DNP":
                return 0.0
            s = str(val)
            if ":" in s:
                parts = s.split(":")
                return int(parts[0]) + int(parts[1]) / 60
            return float(s)
        except Exception:
            return 0.0

    if "MIN_STR" in df.columns:
        df["MIN"] = df["MIN_STR"].apply(parse_minutes)
    else:
        df["MIN"] = 0.0

    # ── Calculate stats — data is already PerGame averages ────────────────────
    # So values like PTS, FGM etc. are already per-game averages from the API
    # We recalculate PIR ourselves from scratch using totals
    # But since we only have PerGame averages, we calculate PIR per game directly

    player_stats = []

    for _, p in df.iterrows():
        gp = float(p.get("GP", 0))
        if gp == 0:
            continue

        pts  = float(p.get("PTS", 0))
        fgm2 = float(p.get("FGM2", 0))
        fga2 = float(p.get("FGA2", 0))
        fgm3 = float(p.get("FGM3", 0))
        fga3 = float(p.get("FGA3", 0))
        fgm  = float(p.get("FGM", 0))
        fga  = float(p.get("FGA", 0))
        ftm  = float(p.get("FTM", 0))
        fta  = float(p.get("FTA", 0))
        treb = float(p.get("TREB", 0))
        ast  = float(p.get("AST", 0))
        stl  = float(p.get("STL", 0))
        tov  = float(p.get("TOV", 0))
        blk  = float(p.get("BLK", 0))
        blka = float(p.get("BLKA", 0))
        pf   = float(p.get("PF", 0))
        fd   = float(p.get("FD", 0))
        mins = float(p.get("MIN", 0))

        # PIR per game (using per-game averages directly)
        missed_fg = fga - fgm
        missed_ft = fta - ftm
        pir_pg = (pts + treb + ast + stl + blk + fd) - (missed_fg + missed_ft + tov + blka + pf)

        # %Min — minutes per game / 40 * 100
        pct_min = safe_div(mins * 100, 40)

        # eFG%
        efg = safe_div((fgm + 0.5 * fgm3) * 100, fga)

        # TS%
        ts = safe_div(pts * 100, 2 * (fga + 0.475 * fta))

        # Shooting %
        fg3_pct = safe_div(fgm3 * 100, fga3)
        fg2_pct = safe_div(fgm2 * 100, fga2)
        ft_pct  = safe_div(ftm * 100, fta)

        player_stats.append({
            "name":    str(p.get("Name", "")),
            "team":    str(p.get("Team", "")),
            "gp":      int(gp),
            "min_pg":  round1(mins),
            "pir":     round1(pir_pg),
            "pct_min": round1(pct_min),
            "efg":     round1(efg),
            "ts":      round1(ts),
            "fg3m_pg": round1(fgm3),
            "fg3a_pg": round1(fga3),
            "fg3_pct": round1(fg3_pct),
            "fg2m_pg": round1(fgm2),
            "fg2a_pg": round1(fga2),
            "fg2_pct": round1(fg2_pct),
            "ftm_pg":  round1(ftm),
            "fta_pg":  round1(fta),
            "ft_pct":  round1(ft_pct),
            "pts_pg":  round1(pts),
            "reb_pg":  round1(treb),
            "ast_pg":  round1(ast),
            "stl_pg":  round1(stl),
            "blk_pg":  round1(blk),
            "tov_pg":  round1(tov),
        })

    return player_stats


# ─────────────────────────────────────────
# STEP 5 — FETCH STANDINGS
# ─────────────────────────────────────────

def fetch_standings(season):
    """
    Calculate standings from schedule metadata.
    The Standings API endpoint returns empty data so we derive from game results.
    """
    try:
        gm = GameMetadata(COMPETITION)
        meta_df = gm.get_gamecodes_season(season)
        if meta_df is None or meta_df.empty:
            return []

        def fc(df, candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        home_col   = fc(meta_df, ["hometeam", "Home", "home"])
        away_col   = fc(meta_df, ["awayteam", "Away", "away"])
        hscore_col = fc(meta_df, ["homescore", "HomeScore", "home_score"])
        ascore_col = fc(meta_df, ["awayscore", "AwayScore", "away_score"])
        played_col = fc(meta_df, ["played", "Played", "IsPlayed"])

        if not all([home_col, away_col, hscore_col, ascore_col, played_col]):
            print(f"  Missing schedule columns for standings. Available: {meta_df.columns.tolist()}")
            return []

        teams = {}
        for _, row in meta_df.iterrows():
            if not row.get(played_col, False):
                continue
            home = row[home_col]
            away = row[away_col]
            try:
                hs  = int(row[hscore_col])
                as_ = int(row[ascore_col])
            except (ValueError, TypeError):
                continue
            if not home or not away:
                continue

            for team, pts_f, pts_a, win in [
                (home, hs, as_, hs > as_),
                (away, as_, hs, as_ > hs),
            ]:
                if team not in teams:
                    teams[team] = {"team": team, "gp": 0, "wins": 0,
                                   "losses": 0, "pts_for": 0, "pts_against": 0}
                teams[team]["gp"]          += 1
                teams[team]["pts_for"]     += pts_f
                teams[team]["pts_against"] += pts_a
                if win:
                    teams[team]["wins"]    += 1
                else:
                    teams[team]["losses"]  += 1

        standings = sorted(
            teams.values(),
            key=lambda t: (t["wins"], t["pts_for"] - t["pts_against"]),
            reverse=True
        )

        for i, t in enumerate(standings):
            gp = t["gp"] or 1
            t["position"]        = i + 1
            t["pct"]             = round(t["wins"] / gp, 3)
            t["diff"]            = t["pts_for"] - t["pts_against"]
            t["pts_for_pg"]      = round(t["pts_for"] / gp, 1)
            t["pts_against_pg"]  = round(t["pts_against"] / gp, 1)
            t["streak"]          = None

        print(f"  Calculated standings for {len(standings)} teams from {sum(t['gp'] for t in standings)//2} games.")
        return standings

    except Exception as e:
        print(f"  ERROR calculating standings: {e}")
        import traceback; traceback.print_exc()
        return []


# ─────────────────────────────────────────
# STEP 6 — WRITE OUTPUT JSON
# ─────────────────────────────────────────

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Wrote {path}")


def run_pipeline(season):
    """Full pipeline for one season."""
    print(f"\n{'='*50}")
    print(f"  Running pipeline for {season_label(season)}")
    print(f"{'='*50}")

    out_dir = os.path.join(DATA_DIR, str(season))
    ensure_dir(out_dir)

    # ── Fetch raw game data ──────────────────────────────────────────────────
    team_games, player_games = fetch_season_games(season)

    if team_games is None:
        print(f"  No data available for {season_label(season)}. Skipping.")
        return

    # ── Calculate stats ──────────────────────────────────────────────────────
    print(f"\n[{season_label(season)}] Calculating team stats...")
    team_stats = calculate_team_stats(team_games)
    print(f"  Calculated stats for {len(team_stats)} teams.")

    print(f"[{season_label(season)}] Calculating player stats...")
    player_stats = calculate_player_stats(player_games) if player_games is not None else []
    print(f"  Calculated stats for {len(player_stats)} players.")

    # ── Fetch schedule and standings ─────────────────────────────────────────
    print(f"[{season_label(season)}] Fetching schedule...")
    schedule = fetch_schedule(season)
    print(f"  Fetched {len(schedule)} games.")

    print(f"[{season_label(season)}] Fetching standings...")
    standings = fetch_standings(season)
    print(f"  Fetched standings for {len(standings)} teams.")

    # ── Write JSON output ────────────────────────────────────────────────────
    print(f"\n[{season_label(season)}] Writing output files...")

    write_json(os.path.join(out_dir, "teams.json"), {
        "season": season,
        "season_label": season_label(season),
        "updated": datetime.now(timezone.utc).isoformat(),
        "teams": team_stats
    })

    write_json(os.path.join(out_dir, "players.json"), {
        "season": season,
        "season_label": season_label(season),
        "updated": datetime.now(timezone.utc).isoformat(),
        "players": player_stats
    })

    write_json(os.path.join(out_dir, "schedule.json"), {
        "season": season,
        "season_label": season_label(season),
        "updated": datetime.now(timezone.utc).isoformat(),
        "games": schedule
    })

    write_json(os.path.join(out_dir, "standings.json"), {
        "season": season,
        "season_label": season_label(season),
        "updated": datetime.now(timezone.utc).isoformat(),
        "standings": standings
    })

    print(f"\n[{season_label(season)}] ✓ Pipeline complete.")


def update_seasons_index(seasons_run):
    """Write/update the master seasons index file."""
    index_path = os.path.join(DATA_DIR, "seasons.json")

    # Load existing if present
    existing = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            existing = json.load(f).get("seasons", [])

    # Merge, deduplicate, sort descending
    all_seasons = sorted(set(existing + seasons_run), reverse=True)

    write_json(index_path, {
        "current_season": CURRENT_SEASON,
        "seasons": all_seasons,
        "season_labels": {s: season_label(s) for s in all_seasons}
    })


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EuroMetrics data pipeline")
    parser.add_argument("--season", type=int, help="Specific season start year to run (e.g. 2024)")
    parser.add_argument("--all", action="store_true", help="Run all historical seasons")
    args = parser.parse_args()

    ensure_dir(DATA_DIR)

    if args.all:
        seasons_to_run = ALL_SEASONS
    elif args.season:
        seasons_to_run = [args.season]
    else:
        seasons_to_run = [CURRENT_SEASON]

    completed = []
    for season in seasons_to_run:
        try:
            run_pipeline(season)
            completed.append(season)
        except Exception as e:
            print(f"\nERROR in season {season_label(season)}: {e}")
            import traceback
            traceback.print_exc()

    if completed:
        update_seasons_index(completed)
        print(f"\n{'='*50}")
        print(f"  All done. Ran {len(completed)} season(s).")
        print(f"{'='*50}\n")
