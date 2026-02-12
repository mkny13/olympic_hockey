"""daily_stats.py

Robust daily boxscore aggregator for the Camelot Caniacs Olympic fantasy league.

Features:
- Uses the NHL stats API (tries a couple base URLs) to fetch schedule and boxscores
- Gathers finished games for yesterday and today, aggregates player stats from boxscores
- Maps stats to players listed in `rosters.csv` and applies weights from `scoring_config.py`
- Persists cumulative totals in `totals.json`
- Writes a sorted `daily_report.txt` and includes a Trash Talk MVP and Watch List for tomorrow

This script is defensive: it tolerates missing fields, empty API responses, and partial matches.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any
import unicodedata
import re
import argparse
import html as _html
try:
    # Python 3.9+ zoneinfo for IANA tz support
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

import requests
try:
    # nhl-api-py wrapper
    from nhlpy import NHLClient
except Exception:
    NHLClient = None

# --- Paths / Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROSTERS_CSV = os.path.join(BASE_DIR, "rosters.csv")
REPORT_TXT = os.path.join(BASE_DIR, "daily_report.txt")
TOTALS_JSON = os.path.join(BASE_DIR, "totals.json")
REPORT_HTML = os.path.join(BASE_DIR, "daily_report.html")

# Try known NHL stats API hosts (prefer api-web.nhle.com per-date endpoints which are reachable
# from this environment; statsapi.web.nhl.com is tried as a fallback)
API_BASES = [
    "https://api-web.nhle.com/v1",
    "https://statsapi.web.nhl.com/api/v1",
]

HEADERS = {"User-Agent": "CamelotCaniacs/1.0 (+https://example)"}


def load_scoring() -> Dict[str, float]:
    """Import scoring settings from scoring_config.py in same folder.

    Assumes `SCORING_SETTINGS` dict is defined.
    """
    try:
        from scoring_config import SCORING_SETTINGS

        return SCORING_SETTINGS
    except Exception as e:
        print(f"⚠️ Couldn't import scoring_config.SCORING_SETTINGS: {e}")
        # Fallback default (keeps previous behavior)
        return {
            "G": 2,
            "A": 1,
            "PPP": 0.5,
            "SOG": 0.1,
            "HIT": 0.1,
            "BLK": 0.5,
            "W": 4,
            "GA": -2,
            "SV": 0.2,
            "SO": 3,
            "OTL": 1,
        }


def load_rosters(path: str) -> Dict[str, Dict[str, str]]:
    """Load `rosters.csv` into dict keyed by normalized player name.

    CSV format expected: Player,FantasyTeam,OlympicTeam
    """
    if not os.path.exists(path):
        print(f"⚠️ Roster file not found at {path}")
        return {}

    rosters: Dict[str, Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row.get("Player") or "").strip()
            if not name:
                continue
            key = _norm(name)
            rosters[key] = {
                "Player": name,
                "FantasyTeam": (row.get("FantasyTeam") or "").strip(),
                "OlympicTeam": (row.get("OlympicTeam") or "").strip(),
            }
    return rosters


def _norm(s: str) -> str:
    # normalize unicode (strip diacritics), remove punctuation, collapse whitespace, lowercase
    if not s:
        return ""
    s2 = unicodedata.normalize("NFKD", s)
    s2 = s2.encode("ascii", "ignore").decode("ascii")
    # remove punctuation (keep alphanum and spaces)
    s2 = re.sub(r"[^A-Za-z0-9\s]", "", s2)
    return " ".join(s2.strip().casefold().split())


def _try_get(url: str, params: dict | None = None, timeout: int = 10) -> Any:
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            # brief backoff
            time.sleep(0.8 + attempt * 0.5)
            last_exc = e
    print(f"⚠️ Request failed for {url} after retries: {last_exc}")
    return None


def _parse_iso_dt(s: str) -> datetime | None:
    """Parse an ISO datetime string (handles trailing Z) into an aware datetime (UTC)."""
    if not s:
        return None
    try:
        # Normalize trailing Z to +00:00 so fromisoformat can parse
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _to_est_str(s: str) -> str:
    """Convert ISO datetime string s -> formatted datetime in America/New_York (EST/EDT).

    Falls back to naive UTC->EST shift (-5h) if zoneinfo unavailable.
    Returns original string if parsing fails.
    """
    dt = _parse_iso_dt(s)
    if not dt:
        return s or ""
    try:
        if ZoneInfo is not None:
            est = dt.astimezone(ZoneInfo("America/New_York"))
            return est.strftime("%Y-%m-%d %H:%M %Z")
        else:
            # naive fallback: assume EST = UTC-5
            est = dt - timedelta(hours=5)
            return est.strftime("%Y-%m-%d %H:%M EST")
    except Exception:
        return s or ""


def _now_est_str() -> str:
    try:
        if ZoneInfo is not None:
            return datetime.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M %Z")
        else:
            # fallback to local now shifted by -5 hours (avoid deprecated utcnow())
            return (datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M EST")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M")


def api_get(path: str, params: dict | None = None) -> Any:
    """Try the configured API_BASES for the given path (path without base)."""
    for base in API_BASES:
        url = base.rstrip("/") + "/" + path.lstrip("/")
        data = _try_get(url, params=params)
        if data:
            return data
    return None


def fetch_schedule_for_dates(start_date: str, end_date: str) -> List[dict]:
    """Fetch schedule between start_date and end_date (YYYY-MM-DD). Returns list of games.

    The NHL schedule JSON usually contains 'dates' -> each with 'games' list.
    """
    games: List[dict] = []
    # Prefer nhl-api-py schedule if available
    try:
        if NHLClient is not None:
            client = NHLClient()
            # iterate dates and call daily_schedule for each
            try:
                s_dt = datetime.fromisoformat(start_date).date()
                e_dt = datetime.fromisoformat(end_date).date()
            except Exception:
                s_dt = datetime.fromisoformat(start_date).date()
                e_dt = s_dt
            cur = s_dt
            while cur <= e_dt:
                try:
                    data = client.schedule.daily_schedule(date=cur.isoformat())
                    if data and isinstance(data, dict):
                        for g in data.get("games", []):
                            games.append(g)
                except Exception:
                    # ignore per-date failures and continue
                    pass
                cur = cur + timedelta(days=1)
            if games:
                return games
    except Exception:
        # if nhlpy isn't usable, fall back to HTTP below
        pass

    # First try the statsapi style (range query)
    path = f"schedule?startDate={start_date}&endDate={end_date}"
    data = api_get(path)
    if data:
        for d in data.get("dates", []):
            for g in d.get("games", []):
                games.append(g)

    # If no results, some NHL endpoints (api-web.nhle.com) use a per-date path: /schedule/{date}
    if not games:
        # iterate each date in the inclusive range and try the per-date endpoint
        try:
            s_dt = datetime.fromisoformat(start_date).date()
            e_dt = datetime.fromisoformat(end_date).date()
        except Exception:
            # fallback: treat as single date
            s_dt = datetime.fromisoformat(start_date).date()
            e_dt = s_dt

        cur = s_dt
        while cur <= e_dt:
            p = f"schedule/{cur.isoformat()}"
            d2 = api_get(p)
            if d2:
                # older/api-web returns a structure with 'gameWeek' or 'dates'
                if isinstance(d2.get("dates"), list):
                    for dd in d2.get("dates", []):
                        for g in dd.get("games", []):
                            games.append(g)
                elif isinstance(d2.get("gameWeek"), list):
                    for gw in d2.get("gameWeek", []):
                        for g in gw.get("games", []):
                            games.append(g)
            cur = cur + timedelta(days=1)

    return games


def fetch_boxscore(game_pk: int) -> dict | None:
    # Prefer the nhl-api-py client (it knows several endpoint shapes) if available
    try:
        if NHLClient is not None:
            try:
                client = NHLClient()
                # the client expects a string game_id
                data = client.game_center.boxscore(game_id=str(game_pk))
                if isinstance(data, dict) and data:
                    return data
            except Exception:
                # Fall back to direct HTTP attempts below
                pass
    except Exception:
        # any unexpected import/runtime errors here shouldn't crash the script
        pass

    # Try several common NHL API paths for per-game boxscore/play-by-play/content
    candidate_paths = [
        f"game/{game_pk}/boxscore",
        f"gamecenter/{game_pk}/boxscore",
        f"gamecenter/{game_pk}/play-by-play",
        f"gamecenter/{game_pk}/content",
        f"game/{game_pk}/content",
        f"game/{game_pk}",
    ]

    for p in candidate_paths:
        data = api_get(p)
        if not data:
            continue
        # Heuristic: a valid box-like payload often contains 'teams' or 'gameData' or 'plays'
        if isinstance(data, dict) and ("teams" in data or "gameData" in data or "plays" in data or "gamePk" in data):
            return data

    return None


def parse_boxscore_for_players(box: dict) -> List[dict]:
    """Extract per-player stat rows from a boxscore, normalized to our scoring keys.

    Returns list of dicts with Player (full name) and stat keys used by scoring.
    """
    rows: List[dict] = []
    if not box:
        return rows

    # Handle nhlpy boxscore shape with 'playerByGameStats'
    pbg = box.get("playerByGameStats")
    if isinstance(pbg, dict) and pbg:
        for side in ("awayTeam", "homeTeam"):
            team_block = pbg.get(side)
            if not team_block:
                continue
            # forwards and defense are skaters
            for role in ("forwards", "defense"):
                for p in team_block.get(role, []) or []:
                    name = (p.get("name") or {}).get("default") or p.get("name") or p.get("playerId")
                    row = {"Player": name, "G": 0, "A": 0, "PPP": 0, "SOG": 0, "HIT": 0, "BLK": 0,
                           "W": 0, "GA": 0, "SV": 0, "SO": 0, "OTL": 0}
                    row["G"] = int(_safe(p.get("goals", 0)))
                    row["A"] = int(_safe(p.get("assists", 0)))
                    pp_g = _safe(p.get("powerPlayGoals", 0))
                    pp_a = _safe(p.get("powerPlayAssists", 0))
                    row["PPP"] = int(pp_g) + int(pp_a)
                    row["SOG"] = int(_safe(p.get("sog", p.get("shots", 0))))
                    row["HIT"] = int(_safe(p.get("hits", 0)))
                    row["BLK"] = int(_safe(p.get("blockedShots", p.get("blocked", 0))))
                    rows.append(row)

            # goalies
            for g in team_block.get("goalies", []) or []:
                name = (g.get("name") or {}).get("default") or g.get("name") or g.get("playerId")
                row = {"Player": name, "G": 0, "A": 0, "PPP": 0, "SOG": 0, "HIT": 0, "BLK": 0,
                       "W": 0, "GA": 0, "SV": 0, "SO": 0, "OTL": 0}
                row["GA"] = int(_safe(g.get("goalsAgainst", g.get("goalsAgainst", 0))))
                row["SV"] = int(_safe(g.get("saves", 0)))
                # decision field may indicate W
                if (g.get("decision") or "").upper() == "W":
                    row["W"] = 1
                # shutout
                if int(row["GA"]) == 0 and _safe(g.get("toi", 0)):
                    row["SO"] = 1
                rows.append(row)

        return rows

    # Determine final scores to identify winners/losers when needed
    try:
        linescore = box.get("gameData", {})
    except Exception:
        linescore = {}

    # The boxscore structure under statsapi has 'teams' -> 'away'/'home' -> 'players'
    teams = box.get("teams") or {}
    # Friendly guard: some API variants return top-level 'teams' nested under 'teams'
    if not teams and isinstance(box.get("teams"), dict):
        teams = box.get("teams")

    for side in ("away", "home"):
        team = teams.get(side)
        if not team:
            continue
        players = team.get("players") or {}
        team_score = team.get("teamStats", {}).get("teamSkaterStats", {}).get("goals")
        # players is mapping keyed by 'IDxxxx'
        for pid, pinfo in players.items():
            person = pinfo.get("person") or {}
            full_name = person.get("fullName") or person.get("fullName") or pid
            stats = pinfo.get("stats") or {}
            # init stat row
            row = {"Player": full_name, "G": 0, "A": 0, "PPP": 0, "SOG": 0, "HIT": 0, "BLK": 0,
                   "W": 0, "GA": 0, "SV": 0, "SO": 0, "OTL": 0}

            # Skater stats
            sk = stats.get("skaterStats")
            if sk:
                row["G"] = int(_safe(sk.get("goals", 0)))
                row["A"] = int(_safe(sk.get("assists", 0)))
                # PPP: use powerPlayGoals + powerPlayAssists if present
                pp_g = _safe(sk.get("powerPlayGoals", 0))
                pp_a = _safe(sk.get("powerPlayAssists", 0))
                row["PPP"] = int(pp_g) + int(pp_a)
                row["SOG"] = int(_safe(sk.get("shots", 0)))
                row["HIT"] = int(_safe(sk.get("hits", 0)))
                row["BLK"] = int(_safe(sk.get("blocked", sk.get("blockedShots", 0))))

            # Goalie stats
            gk = stats.get("goalieStats")
            if gk:
                # Saves: saves may be provided directly
                row["GA"] = int(_safe(gk.get("goalsAgainst", 0)))
                # Some variants provide saves or shotsAgainst; compute if necessary
                saves = gk.get("saves")
                if saves is None:
                    shots_against = _safe(gk.get("shotsAgainst", 0))
                    row["SV"] = int(shots_against) - int(row["GA"]) if shots_against is not None else 0
                else:
                    row["SV"] = int(_safe(saves))

                # shutout if goalsAgainst == 0 and timeOnIce > 0
                if int(row["GA"]) == 0 and _safe(gk.get("timeOnIce", "0:00")):
                    row["SO"] = 1

                # Decision (W/L/OT) - some APIs include 'decision' field; otherwise approximate
                decision = gk.get("decision") or ""
                if decision.upper() == "W":
                    row["W"] = 1
                elif decision.upper() in ("L", "" ):
                    # if no explicit decision, we'll decide below using final scores
                    pass

            rows.append(row)

    # Try to mark wins/loss/OTL for goalies by using the game final score if available
    try:
        # navigate to score info depending on variant
        # one variant: box['teams']['away']['team']['id'] and team['goals'] under 'teamStats' sometimes
        # fallback: look up gamePk -> use feed live, but keep simple: if team present with 'goals' in team object
        pass
    except Exception:
        pass

    return rows


def _safe(v, default=0):
    try:
        if v is None:
            return default
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v
    except Exception:
        return default


def aggregate_by_player(rows: List[dict]) -> Dict[str, dict]:
    """Sum stats for players by normalized name. Returns mapping norm_name -> aggregated row."""
    agg: Dict[str, dict] = {}
    for r in rows:
        name = r.get("Player") or ""
        key = _norm(name)
        if key not in agg:
            agg[key] = {k: 0 for k in ("G", "A", "PPP", "SOG", "HIT", "BLK", "W", "GA", "SV", "SO", "OTL")}
            agg[key]["Player"] = name
        for k in ("G", "A", "PPP", "SOG", "HIT", "BLK", "W", "GA", "SV", "SO", "OTL"):
            agg[key][k] = agg[key].get(k, 0) + int(_safe(r.get(k, 0)))
    return agg


def load_totals(path: str) -> Dict[str, float]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def save_totals(path: str, data: Dict[str, float]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Failed to save totals: {e}")


def compute_points_for_row(row: dict, scoring: Dict[str, float]) -> float:
    pts = 0.0
    for k, w in scoring.items():
        pts += float(row.get(k, 0)) * float(w)
    return pts


def generate_report(roster_map: Dict[str, dict], per_player: Dict[str, dict], scoring: Dict[str, float], date_window: str, games_in_window: List[dict], next_window_games: List[dict], next_window_range: str):
    # Load cumulative totals
    totals = load_totals(TOTALS_JSON)

    # Build list of rostered players who have stats
    players_data = []  # list of dicts with Player, FantasyTeam, OlympicTeam, stats, DailyPts, TotalPts
    for norm, stats in per_player.items():
        # Try exact match first, otherwise fall back to unique last-name match
        match_key = norm if norm in roster_map else None
        if not match_key:
            last = norm.split()[-1] if norm.split() else ""
            candidates = [k for k in roster_map.keys() if k.split() and k.split()[-1] == last]
            if len(candidates) == 1:
                match_key = candidates[0]

        if not match_key:
            # no roster match found; skip
            continue

        r = roster_map[match_key]
        daily_pts = compute_points_for_row(stats, scoring)
        prev = totals.get(match_key, 0.0)
        total_pts = float(prev) + float(daily_pts)
        totals[match_key] = total_pts
        players_data.append({
            "Norm": match_key,
            "Player": r["Player"],
            "FantasyTeam": r["FantasyTeam"] or "(No Team)",
            "OlympicTeam": r["OlympicTeam"],
            "Stats": stats,
            "DailyPts": daily_pts,
            "TotalPts": total_pts,
        })

    # Save updated totals
    save_totals(TOTALS_JSON, totals)

    # Summarize by team
    teams: Dict[str, List[dict]] = defaultdict(list)
    for p in players_data:
        teams[p["FantasyTeam"]].append(p)

    # Compute team totals
    team_summary = []
    for team, members in teams.items():
        team_total = sum(m["DailyPts"] for m in members)
        team_summary.append({"FantasyTeam": team, "TeamDailyPts": team_total})

    # Sort: Team Total Desc, then Team Name Asc
    team_summary.sort(key=lambda x: (-x["TeamDailyPts"], x["FantasyTeam"]))

    # Within each team, sort players by DailyPts desc
    for team in teams:
        teams[team].sort(key=lambda m: (-m["DailyPts"], m["Player"]))

    # Compose report text
    lines: List[str] = []
    header = "═══ CAMELOT CANIACS OLYMPIC UPDATE ═══"
    lines.append(header)
    # date_window already includes dates; add times if possible by inspecting games_in_window
    window_times = date_window
    try:
        starts = []
        for g in games_in_window:
            st = g.get("startTimeUTC") or g.get("startTime") or g.get("start_time")
            if st:
                try:
                    dt = _parse_iso_dt(st)
                    if dt:
                        starts.append(dt)
                except Exception:
                    pass
        if starts:
            min_st = min(starts)
            max_st = max(starts)
            window_times = f"{_to_est_str(min_st.isoformat())} to {_to_est_str(max_st.isoformat())}"
    except Exception:
        pass

    lines.append(f"Window: {window_times}")
    lines.append(f"Generated: {_now_est_str()}")
    lines.append("")

    # List games played during the window (finished games with scores)
    lines.append("GAMES PLAYED IN WINDOW:")
    if games_in_window:
        for g in games_in_window:
            try:
                # support multiple shapes
                home = g.get("homeTeam") or g.get("home") or {}
                away = g.get("awayTeam") or g.get("away") or {}
                home_abbrev = (home.get("abbrev") or (home.get("team") or {}).get("abbrev") or home.get("commonName") or "Home")
                away_abbrev = (away.get("abbrev") or (away.get("team") or {}).get("abbrev") or away.get("commonName") or "Away")
                home_score = (home.get("score") or (home.get("team") or {}).get("score") or '')
                away_score = (away.get("score") or (away.get("team") or {}).get("score") or '')
                st = g.get("startTimeUTC") or g.get("startTime") or g.get("start_time") or ""
                st_fmt = _to_est_str(st)
                status = (g.get("gameState") or (g.get("status") or {}).get("detailedState") or "").upper()
                if home_score != '' or away_score != '':
                    lines.append(f" - {away_abbrev} @ {home_abbrev} — {away_score} - {home_score} ({status}) @ {st_fmt}")
                else:
                    lines.append(f" - {away_abbrev} @ {home_abbrev} — {status} @ {st_fmt}")
            except Exception:
                continue
    else:
        lines.append(" - No games found in window or schedule unavailable.")

    # Team daily totals section
    lines.append("")
    lines.append("TEAM DAILY TOTALS:")

    for team_entry in team_summary:
        team = team_entry["FantasyTeam"]
        t_total = team_entry["TeamDailyPts"]
        lines.append(f"{team} — Team Daily: {t_total:.1f}")
        lines.append("-" * 60)
        # header for player lines
        lines.append(f"{'Player':25} | {'Stats':28} | Daily | Total")
        for p in teams[team]:
            s = p["Stats"]
            # representation: for skaters show G-A-PPP SOG, for goalies prefer W-SV-GA
            if p["Stats"].get("SV", 0) > 0 or p["Stats"].get("GA", 0) > 0:
                stat_str = f"W:{int(s.get('W',0))} SV:{int(s.get('SV',0))} GA:{int(s.get('GA',0))}"
            else:
                stat_str = f"{int(s.get('G',0))}G {int(s.get('A',0))}A PPP:{int(s.get('PPP',0))} SOG:{int(s.get('SOG',0))}"
            lines.append(f"{p['Player'][:25]:25} | {stat_str:28} | {p['DailyPts']:5.1f} | {p['TotalPts']:5.1f}")
        lines.append("")

    # Top 5 scorers across rostered players (by DailyPts)
    lines.append("")
    lines.append("TOP 5 SCORERS:")
    top5 = sorted(players_data, key=lambda x: -x["DailyPts"])[:5]
    if top5:
        for p in top5:
            lines.append(f" - {p['Player']} ({p['FantasyTeam']}) — {p['DailyPts']:.1f} FP")
    else:
        lines.append(" - No scorers today.")

    # Next window games
    lines.append("")
    lines.append(f"UPCOMING GAMES: {next_window_range}")
    if next_window_games:
        for g in next_window_games:
            try:
                home = g.get("homeTeam") or g.get("home") or {}
                away = g.get("awayTeam") or g.get("away") or {}
                home_abbrev = (home.get("abbrev") or (home.get("team") or {}).get("abbrev") or home.get("commonName") or "Home")
                away_abbrev = (away.get("abbrev") or (away.get("team") or {}).get("abbrev") or away.get("commonName") or "Away")
                st = g.get("startTimeUTC") or g.get("startTime") or g.get("start_time") or ""
                st_fmt = _to_est_str(st)
                lines.append(f" - {away_abbrev} @ {home_abbrev} — {st_fmt}")
            except Exception:
                continue
    else:
        lines.append(" - No scheduled games found for next window.")

    # Standings: total scores across the 2026 Olympics for each FantasyTeam (sum of roster totals)
    lines.append("")
    lines.append("STANDINGS (Fantasy team totals - 2026):")
    # Build FantasyTeam totals from roster_map and persisted totals
    fantasy_scores: Dict[str, float] = defaultdict(float)
    for norm_key, info in roster_map.items():
        ft = (info.get("FantasyTeam") or "").strip() or "(No Team)"
        if not ft:
            continue
        fantasy_scores[ft] += float(totals.get(norm_key, 0.0))

    if fantasy_scores:
        sorted_fant = sorted(fantasy_scores.items(), key=lambda x: -x[1])
        for team_name, pts in sorted_fant:
            lines.append(f" - {team_name}: {pts:.1f} FP")
    else:
        lines.append(" - No Fantasy standings available (no roster totals found).")

    # Write report
    try:
        with open(REPORT_TXT, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
    except Exception as e:
        print(f"⚠️ Could not write report to {REPORT_TXT}: {e}")

    # Also write a nicer HTML version (structured with tables and CSS)
    try:
        esc = _html.escape

        def esc_txt(v):
            return esc(str(v)) if v is not None else ""

        html_parts = []
        html_parts.append("<!doctype html>")
        html_parts.append('<html><head><meta charset="utf-8"><title>Camelot Caniacs Daily Report</title>')
        css = """<style>
            body{font-family:Inter, Roboto, -apple-system, system-ui, 'Segoe UI', Arial, sans-serif; margin:18px; color:#111}
            h1{font-size:18px;margin-bottom:6px}
            h2{font-size:14px;margin:12px 0 6px}
            table{border-collapse:collapse;width:100%;margin-bottom:12px}
            th,td{border:1px solid #eee;padding:6px 8px;text-align:left;font-size:13px}
            th{background:#f7f7f7;font-weight:600}
            .team-header{background:#222;color:#fff;padding:8px 10px;margin-top:10px;border-radius:4px}
            .muted{color:#666;font-size:12px}
            .monospace{font-family:monospace}
        </style>"""
        html_parts.append(css)
        html_parts.append('</head><body>')
        html_parts.append(f'<h1>CAMELOT CANIACS — Olympic Update</h1>')
        html_parts.append(f'<div class="muted">Window: {esc_txt(window_times)} &nbsp; | &nbsp; Generated: {esc_txt(_now_est_str())}</div>')

        # Games played
        html_parts.append('<h2>Games Played in Window</h2>')
        if games_in_window:
            html_parts.append('<table><thead><tr><th>Away</th><th>Home</th><th>Score</th><th>Status</th><th>Start (EST)</th></tr></thead><tbody>')
            for g in games_in_window:
                try:
                    home = g.get("homeTeam") or g.get("home") or {}
                    away = g.get("awayTeam") or g.get("away") or {}
                    home_abbrev = esc_txt(home.get("abbrev") or (home.get("team") or {}).get("abbrev") or home.get("commonName") or "Home")
                    away_abbrev = esc_txt(away.get("abbrev") or (away.get("team") or {}).get("abbrev") or away.get("commonName") or "Away")
                    home_score = esc_txt(home.get("score") or (home.get("team") or {}).get("score") or "")
                    away_score = esc_txt(away.get("score") or (away.get("team") or {}).get("score") or "")
                    st = g.get("startTimeUTC") or g.get("startTime") or g.get("start_time") or ""
                    st_fmt = esc_txt(_to_est_str(st))
                    status = esc_txt((g.get("gameState") or (g.get("status") or {}).get("detailedState") or "").upper())
                    score_cell = (f"{away_score} - {home_score}" if (home_score != '' or away_score != '') else "")
                    html_parts.append(f'<tr><td>{away_abbrev}</td><td>{home_abbrev}</td><td>{score_cell}</td><td>{status}</td><td class="monospace">{st_fmt}</td></tr>')
                except Exception:
                    continue
            html_parts.append('</tbody></table>')
        else:
            html_parts.append('<div class="muted">No games found in window or schedule unavailable.</div>')

        # Team daily totals
        html_parts.append('<h2>Team Daily Totals</h2>')
        for team_entry in team_summary:
            team = team_entry["FantasyTeam"]
            t_total = team_entry["TeamDailyPts"]
            html_parts.append(f'<div class="team-header">{esc_txt(team)} — Team Daily: {t_total:.1f}</div>')
            html_parts.append('<table><thead><tr><th>Player</th><th>Stats</th><th>Daily</th><th>Total</th></tr></thead><tbody>')
            for p in teams[team]:
                s = p["Stats"]
                if p["Stats"].get("SV", 0) > 0 or p["Stats"].get("GA", 0) > 0:
                    stat_str = f"W:{int(s.get('W',0))} SV:{int(s.get('SV',0))} GA:{int(s.get('GA',0))}"
                else:
                    stat_str = f"{int(s.get('G',0))}G {int(s.get('A',0))}A PPP:{int(s.get('PPP',0))} SOG:{int(s.get('SOG',0))}"
                html_parts.append(f'<tr><td>{esc_txt(p["Player"])}</td><td class="monospace">{esc_txt(stat_str)}</td><td>{p["DailyPts"]:.1f}</td><td>{p["TotalPts"]:.1f}</td></tr>')
            html_parts.append('</tbody></table>')

        # Top 5 scorers
        html_parts.append('<h2>Top 5 Scorers</h2>')
        html_parts.append('<ol>')
        for p in top5:
            html_parts.append(f'<li>{esc_txt(p["Player"])} ({esc_txt(p["FantasyTeam"])}) — {p["DailyPts"]:.1f} FP</li>')
        html_parts.append('</ol>')

        # Upcoming games
        html_parts.append(f'<h2>Upcoming Games: {esc_txt(next_window_range)}</h2>')
        if next_window_games:
            html_parts.append('<table><thead><tr><th>Away</th><th>Home</th><th>Start (EST)</th></tr></thead><tbody>')
            for g in next_window_games:
                try:
                    home = g.get("homeTeam") or g.get("home") or {}
                    away = g.get("awayTeam") or g.get("away") or {}
                    home_abbrev = esc_txt(home.get("abbrev") or (home.get("team") or {}).get("abbrev") or home.get("commonName") or "Home")
                    away_abbrev = esc_txt(away.get("abbrev") or (away.get("team") or {}).get("abbrev") or away.get("commonName") or "Away")
                    st = g.get("startTimeUTC") or g.get("startTime") or g.get("start_time") or ""
                    st_fmt = esc_txt(_to_est_str(st))
                    html_parts.append(f'<tr><td>{away_abbrev}</td><td>{home_abbrev}</td><td class="monospace">{st_fmt}</td></tr>')
                except Exception:
                    continue
            html_parts.append('</tbody></table>')
        else:
            html_parts.append('<div class="muted">No scheduled games found for next window.</div>')

        # Standings by fantasy teams
        html_parts.append('<h2>Standings (Fantasy team totals - 2026)</h2>')
        if fantasy_scores:
            html_parts.append('<table><thead><tr><th>Fantasy Team</th><th>Points</th></tr></thead><tbody>')
            for team_name, pts in sorted_fant:
                html_parts.append(f'<tr><td>{esc_txt(team_name)}</td><td>{pts:.1f}</td></tr>')
            html_parts.append('</tbody></table>')
        else:
            html_parts.append('<div class="muted">No Fantasy standings available (no roster totals found).</div>')

        html_parts.append('</body></html>')
        html_doc = '\n'.join(html_parts)
        with open(REPORT_HTML, "w", encoding="utf-8") as fh:
            fh.write(html_doc)
    except Exception as e:
        print(f"⚠️ Could not write HTML report to {REPORT_HTML}: {e}")

    print("Report written to:", REPORT_TXT)
    print("HTML report written to:", REPORT_HTML)


def _simple_team_key(s: str) -> str:
    """Normalize team/country name to a simple lowercase token for matching (e.g. 'CAN' or 'Canada' -> 'can')."""
    return s.strip().casefold().replace(" ", "")


def main():
    print("🏒 Camelot Caniacs - daily_stats running...")
    scoring = load_scoring()
    roster_map = load_rosters(ROSTERS_CSV)

    # Build date window: yesterday + today
    try:
        if ZoneInfo is not None:
            today = datetime.now(tz=ZoneInfo("UTC")).date()
        else:
            today = datetime.now().date()
    except Exception:
        today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    start = yesterday.isoformat()
    end = today.isoformat()
    date_window = f"{start} to {end}"
    
    # CLI: optional target date (YYYY-MM-DD). Default to yesterday's date.
    parser = argparse.ArgumentParser(description="Generate daily fantasy report for a specific date (defaults to yesterday).")
    parser.add_argument("-d", "--date", help="Target date in YYYY-MM-DD (defaults to yesterday)")
    ns = parser.parse_args()

    # determine target_date (date object)
    if ns.date:
        try:
            target_date = datetime.fromisoformat(ns.date).date()
        except Exception:
            print(f"⚠️ Invalid date format: {ns.date}. Expected YYYY-MM-DD.")
            return
    else:
        target_date = yesterday

    start = target_date.isoformat()
    end = target_date.isoformat()
    date_window = start

    # Fetch schedule and find finished games
    games = fetch_schedule_for_dates(start, end)
    finished_game_pks: List[int] = []
    for g in games:
        try:
            # Handle multiple schedule shapes: some have 'status' dict, others have 'gameState' and 'id'
            is_final = False
            # status.variant
            if isinstance(g.get("status"), dict):
                status = g.get("status", {})
                detailed = (status.get("detailedState") or "").lower()
                abstract = (status.get("abstractGameState") or "").lower()
                if "final" in detailed or abstract == "final":
                    is_final = True
            else:
                gs = (g.get("gameState") or g.get("game_state") or "").lower()
                if gs and "final" in gs:
                    is_final = True

            if is_final:
                pk = g.get("gamePk") or g.get("gameId") or g.get("id") or g.get("game_id")
                try:
                    pk = int(pk)
                except Exception:
                    pk = 0
                if pk:
                    finished_game_pks.append(pk)
        except Exception:
            continue

    if not finished_game_pks:
        print("⚠️ No finished games found for yesterday/today window.")

    # For each finished game, fetch boxscore and parse
    all_rows: List[dict] = []
    for pk in finished_game_pks:
        box = fetch_boxscore(pk)
        rows = parse_boxscore_for_players(box)
        all_rows.extend(rows)

    # Aggregate per player
    per_player = aggregate_by_player(all_rows)

    # Get tomorrow schedule for next-window games
    next_day = target_date + timedelta(days=1)
    tomorrow_games = fetch_schedule_for_dates(next_day.isoformat(), next_day.isoformat())

    # Build list of game dicts that were finished in the window
    finished_games = []
    finished_ids = set(finished_game_pks)
    for g in games:
        gid = g.get("gamePk") or g.get("gameId") or g.get("id") or g.get("game_id")
        try:
            if int(gid) in finished_ids:
                finished_games.append(g)
        except Exception:
            continue

    next_window_range = next_day.isoformat()

    # Generate report (also persists totals)
    generate_report(roster_map, per_player, scoring, date_window, finished_games, tomorrow_games, next_window_range)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}")
        sys.exit(1)
