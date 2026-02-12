"""daily_stats.py

Robust daily boxscore aggregator for the Camelot Caniacs Olympic fantasy league.
Updates: Now tracks processed_game_ids to prevent double-counting on multiple runs.
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
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

import requests
try:
    from nhlpy import NHLClient
except Exception:
    NHLClient = None

# --- Paths / Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROSTERS_CSV = os.path.join(BASE_DIR, "rosters.csv")
REPORT_TXT = os.path.join(BASE_DIR, "daily_report.txt")
TOTALS_JSON = os.path.join(BASE_DIR, "totals.json")
REPORT_HTML = os.path.join(BASE_DIR, "daily_report.html")

API_BASES = [
    "https://api-web.nhle.com/v1",
    "https://statsapi.web.nhl.com/api/v1",
]

HEADERS = {"User-Agent": "CamelotCaniacs/1.0"}

def load_scoring() -> Dict[str, float]:
    try:
        from scoring_config import SCORING_SETTINGS
        return SCORING_SETTINGS
    except Exception:
        return {"G": 2, "A": 1, "PPP": 0.5, "SOG": 0.1, "HIT": 0.1, "BLK": 0.5, "W": 4, "GA": -2, "SV": 0.2, "SO": 3, "OTL": 1}

def load_rosters(path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(path): return {}
    rosters = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row.get("Player") or "").strip()
            if not name: continue
            key = _norm(name)
            rosters[key] = {
                "Player": name,
                "FantasyTeam": (row.get("FantasyTeam") or "").strip(),
                "OlympicTeam": (row.get("OlympicTeam") or "").strip(),
            }
    return rosters

def _norm(s: str) -> str:
    if not s: return ""
    s2 = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s2 = re.sub(r"[^A-Za-z0-9\s]", "", s2)
    return " ".join(s2.strip().casefold().split())

def _try_get(url: str, params: dict | None = None) -> Any:
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(0.5)
    return None

def api_get(path: str) -> Any:
    for base in API_BASES:
        url = base.rstrip("/") + "/" + path.lstrip("/")
        data = _try_get(url)
        if data: return data
    return None

def fetch_schedule_for_dates(start_date: str, end_date: str) -> List[dict]:
    games = []
    # Try HTTP schedule first
    path = f"schedule/{start_date}"
    data = api_get(path)
    if data:
        if isinstance(data.get("gameWeek"), list):
            for gw in data.get("gameWeek", []):
                for g in gw.get("games", []):
                    games.append(g)
        elif isinstance(data.get("dates"), list):
             for d in data.get("dates", []):
                for g in d.get("games", []):
                    games.append(g)
    return games

def fetch_boxscore(game_pk: int) -> dict | None:
    return api_get(f"gamecenter/{game_pk}/boxscore") or api_get(f"game/{game_pk}/boxscore")

def parse_boxscore_for_players(box: dict) -> List[dict]:
    rows = []
    if not box: return rows
    
    # 1. Try "playerByGameStats" (New API Format)
    pbg = box.get("playerByGameStats")
    if pbg:
        for side in ("awayTeam", "homeTeam"):
            team = pbg.get(side, {})
            # Skaters
            for role in ("forwards", "defense"):
                for p in team.get(role, []) or []:
                    name = (p.get("name") or {}).get("default") or p.get("name")
                    row = {"Player": name, "G":0, "A":0, "PPP":0, "SOG":0, "HIT":0, "BLK":0, "W":0, "GA":0, "SV":0, "SO":0, "OTL":0}
                    row["G"] = p.get("goals", 0)
                    row["A"] = p.get("assists", 0)
                    row["PPP"] = p.get("powerPlayGoals", 0) + p.get("powerPlayAssists", 0)
                    row["SOG"] = p.get("shots", 0)
                    row["HIT"] = p.get("hits", 0)
                    row["BLK"] = p.get("blockedShots", 0)
                    rows.append(row)
            # Goalies
            for g in team.get("goalies", []) or []:
                name = (g.get("name") or {}).get("default") or g.get("name")
                row = {"Player": name, "G":0, "A":0, "PPP":0, "SOG":0, "HIT":0, "BLK":0, "W":0, "GA":0, "SV":0, "SO":0, "OTL":0}
                row["GA"] = g.get("goalsAgainst", 0)
                row["SV"] = g.get("saves", 0)
                
                # Decision: Check explicitly for "W"
                if (g.get("decision") or "").upper() == "W": 
                    row["W"] = 1
                
                # ROBUST SHUTOUT LOGIC:
                # If you Won, allowed 0 goals, and made at least 1 save, it's a Shutout.
                # This ignores 'toi' parsing issues.
                if row["GA"] == 0 and row["W"] == 1 and row["SV"] > 0:
                    row["SO"] = 1
                
                rows.append(row)

    # 2. Fallback for "boxscore" / "teams" (Old API Format)
    elif "teams" in box:
        teams = box.get("teams")
        for side in ("away", "home"):
            team = teams.get(side) or {}
            players = team.get("players") or {}
            for pid, pinfo in players.items():
                person = pinfo.get("person") or {}
                name = person.get("fullName") or "Unknown"
                stats = (pinfo.get("stats") or {}).get("skaterStats")
                goalie_stats = (pinfo.get("stats") or {}).get("goalieStats")
                
                row = {"Player": name, "G":0, "A":0, "PPP":0, "SOG":0, "HIT":0, "BLK":0, "W":0, "GA":0, "SV":0, "SO":0, "OTL":0}
                
                if stats:
                    row["G"] = stats.get("goals", 0)
                    row["A"] = stats.get("assists", 0)
                    row["PPP"] = stats.get("powerPlayGoals", 0) + stats.get("powerPlayAssists", 0)
                    row["SOG"] = stats.get("shots", 0)
                    row["HIT"] = stats.get("hits", 0)
                    row["BLK"] = stats.get("blocked", 0)
                    rows.append(row)
                
                elif goalie_stats:
                    row["GA"] = goalie_stats.get("goalsAgainst", 0)
                    row["SV"] = goalie_stats.get("saves", 0)
                    # Old API decision is sometimes missing, check stats
                    if (goalie_stats.get("decision") or "").upper() == "W":
                         row["W"] = 1
                    
                    # Robust Shutout Check
                    if row["GA"] == 0 and row["W"] == 1 and row["SV"] > 0:
                        row["SO"] = 1
                    rows.append(row)
                    
    return rows

def aggregate_by_player(rows: List[dict]) -> Dict[str, dict]:
    agg = {}
    for r in rows:
        name = r.get("Player") or ""
        key = _norm(name)
        if key not in agg:
            agg[key] = {k: 0 for k in ("G", "A", "PPP", "SOG", "HIT", "BLK", "W", "GA", "SV", "SO", "OTL")}
            agg[key]["Player"] = name
        for k in agg[key]:
            if k != "Player":
                agg[key][k] += int(r.get(k, 0))
    return agg

# --- NEW: Totals Management with Game ID Tracking ---
def load_totals_data(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                # Migration: if old format (flat dict), convert to new format
                if data and not isinstance(data.get("scores"), dict):
                    return {"scores": data, "processed_games": []}
                return data
        except Exception:
            pass
    return {"scores": {}, "processed_games": []}

def save_totals_data(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

def compute_points(stats: dict, scoring: dict) -> float:
    return sum(float(stats.get(k, 0)) * float(scoring.get(k, 0)) for k in scoring)

def generate_report(roster, scoring, date_window, finished_games, boxscore_map):
    # Load persistence
    data = load_totals_data(TOTALS_JSON)
    total_scores = data.get("scores", {})
    processed_ids = set(data.get("processed_games", []))
    
    # 1. Calculate stats for the REPORT (All games in window)
    all_window_rows = []
    for pk, rows in boxscore_map.items():
        all_window_rows.extend(rows)
    daily_stats = aggregate_by_player(all_window_rows)
    
    # 2. Calculate stats for TOTALS (Only NEW games)
    new_rows = []
    new_game_ids = []
    for pk, rows in boxscore_map.items():
        if pk not in processed_ids:
            new_rows.extend(rows)
            new_game_ids.append(pk)
    new_stats = aggregate_by_player(new_rows)
    
    # 3. Update Totals
    for norm, stats in new_stats.items():
        # Match to roster
        match_key = norm if norm in roster else None
        if not match_key:
             # simple last name fallback
            last = norm.split()[-1] if norm.split() else ""
            cands = [k for k in roster.keys() if k.endswith(f" {last}") or k == last]
            if len(cands) == 1: match_key = cands[0]
        
        if match_key:
            pts = compute_points(stats, scoring)
            total_scores[match_key] = total_scores.get(match_key, 0.0) + pts

    # Save new state
    data["scores"] = total_scores
    data["processed_games"] = list(processed_ids.union(set(new_game_ids)))
    save_totals_data(TOTALS_JSON, data)
    
    # 4. Build Output Data
    players_output = []
    for norm, stats in daily_stats.items():
        # Roster Match
        match_key = norm if norm in roster else None
        if not match_key:
            last = norm.split()[-1] if norm.split() else ""
            cands = [k for k in roster.keys() if k.endswith(f" {last}") or k == last]
            if len(cands) == 1: match_key = cands[0]
            
        if match_key:
            r = roster[match_key]
            d_pts = compute_points(stats, scoring)
            t_pts = total_scores.get(match_key, 0.0)
            players_output.append({
                "Player": r["Player"],
                "FantasyTeam": r["FantasyTeam"],
                "Stats": stats,
                "DailyPts": d_pts,
                "TotalPts": t_pts
            })

    # Sort and Write Report
    write_text_report(players_output, finished_games, date_window)
    write_html_report(players_output, finished_games, date_window)
    push_to_github()

def write_text_report(players, games, window):
    lines = ["═══ CAMELOT CANIACS OLYMPIC UPDATE ═══", f"Window: {window}\n"]
    lines.append("GAMES PLAYED:")
    for g in games:
        # Simplified game line
        lines.append(f" - {g.get('awayTeam',{}).get('abbrev')} @ {g.get('homeTeam',{}).get('abbrev')} (FINAL)")
    
    # Team Grouping
    teams = defaultdict(list)
    for p in players: teams[p["FantasyTeam"]].append(p)
    
    # Sort Teams by Total Daily Pts
    team_sums = sorted([(t, sum(x['DailyPts'] for x in m)) for t, m in teams.items()], key=lambda x: (-x[1], x[0]))
    
    lines.append("\nTEAM DAILY TOTALS:")
    for team, total in team_sums:
        lines.append(f"{team} — Daily: {total:.1f}")
        lines.append("-" * 65)
        lines.append(f"{'Player':20} | {'Stats':25} | Daily | Total")
        # Sort players by daily pts
        for p in sorted(teams[team], key=lambda x: -x['DailyPts']):
            s = p['Stats']
            if s.get('SV') or s.get('GA'):
                stat = f"W:{s['W']} SV:{s['SV']} GA:{s['GA']}"
            else:
                stat = f"{s['G']}G {s['A']}A SOG:{s['SOG']}"
            lines.append(f"{p['Player'][:20]:20} | {stat:25} | {p['DailyPts']:5.1f} | {p['TotalPts']:5.1f}")
        lines.append("")
        
    with open(REPORT_TXT, "w") as f: f.write("\n".join(lines))
    print(f"Report written to {REPORT_TXT}")

def write_html_report(players, games, window):
    # Basic HTML structure
    html = f"""<html><head><style>
    body{{font-family:sans-serif; padding:20px}} table{{border-collapse:collapse; width:100%}} 
    th,td{{border:1px solid #ddd; padding:8px; text-align:left}} th{{background-color:#f2f2f2}}
    .team{{background:#333; color:white; padding:10px; margin-top:20px}}
    </style></head><body><h1>Camelot Caniacs Update</h1><p>{window}</p>"""
    
    teams = defaultdict(list)
    for p in players: teams[p["FantasyTeam"]].append(p)
    team_sums = sorted([(t, sum(x['DailyPts'] for x in m)) for t, m in teams.items()], key=lambda x: (-x[1], x[0]))
    
    for team, total in team_sums:
        html += f"<div class='team'><b>{team}</b> - Daily: {total:.1f}</div><table><tr><th>Player</th><th>Stats</th><th>Daily</th><th>Total</th></tr>"
        for p in sorted(teams[team], key=lambda x: -x['DailyPts']):
             s = p['Stats']
             stat = f"W:{s['W']} SV:{s['SV']} GA:{s['GA']}" if (s.get('SV') or s.get('GA')) else f"{s['G']}G {s['A']}A {s['SOG']}SOG"
             html += f"<tr><td>{p['Player']}</td><td>{stat}</td><td>{p['DailyPts']:.1f}</td><td>{p['TotalPts']:.1f}</td></tr>"
        html += "</table>"
    
    html += "</body></html>"
    with open(REPORT_HTML, "w") as f: f.write(html)

def push_to_github():
    import subprocess
    try:
        os.chdir(BASE_DIR)
        subprocess.run(["git", "add", "daily_report.txt", "daily_report.html", "totals.json"], check=True)
        subprocess.run(["git", "commit", "-m", f"Update: {datetime.now().strftime('%Y-%m-%d')}"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("🚀 GitHub updated successfully!")
    except Exception as e:
        print(f"⚠️ Git push failed: {e}")

def main():
    print("🏒 Camelot Caniacs - daily_stats running...")
    scoring = load_scoring()
    roster = load_rosters(ROSTERS_CSV)
    
    # Defaults to Yesterday's games
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    games = fetch_schedule_for_dates(yesterday, yesterday)
    
    # Identify finished games
    finished_games = []
    finished_ids = []
    for g in games:
        # Check for Final state
        state = (g.get("gameState") or g.get("status", {}).get("detailedState") or "").upper()
        if "FINAL" in state:
            finished_games.append(g)
            finished_ids.append(g.get("gamePk") or g.get("id"))

    if not finished_games:
        print("⚠️ No finished games found for yesterday.")
        # proceed anyway to generate empty report if needed, or exit
        
    # Fetch boxscores
    boxscore_map = {}
    for pid in finished_ids:
        boxscore_map[pid] = parse_boxscore_for_players(fetch_boxscore(pid))
        
    generate_report(roster, scoring, f"Date: {yesterday}", finished_games, boxscore_map)

if __name__ == "__main__":
    main()