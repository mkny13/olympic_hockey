"""daily_stats.py

Robust daily boxscore aggregator for the Camelot Caniacs Olympic fantasy league.
Updates: 
- Fixes bug where Daily Points were 0 despite Totals updating (Logic Sync).
- Ensures Report and Totals use identical player matching logic.
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
import requests
import argparse

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

def fetch_schedule_for_date(target_date: str) -> List[dict]:
    games = []
    path = f"schedule/{target_date}"
    data = api_get(path)
    
    if data:
        if isinstance(data.get("gameWeek"), list):
            for day_obj in data["gameWeek"]:
                if day_obj.get("date") == target_date:
                    games.extend(day_obj.get("games", []))
        elif isinstance(data.get("dates"), list):
            for day_obj in data["dates"]:
                if day_obj.get("date") == target_date:
                    games.extend(day_obj.get("games", []))
    return games

def fetch_boxscore(game_pk: int) -> dict | None:
    return api_get(f"gamecenter/{game_pk}/boxscore") or api_get(f"game/{game_pk}/boxscore")

def parse_boxscore_for_players(box: dict) -> List[dict]:
    rows = []
    if not box: return rows
    
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
                
                decision = (g.get("decision") or "").upper()
                if decision == "W": 
                    row["W"] = 1
                elif decision in ("OT", "OTL"):
                    row["OTL"] = 1
                
                if row["GA"] == 0 and row["W"] == 1 and row["SV"] > 0:
                    row["SO"] = 1
                rows.append(row)

    # Fallback for Old API
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
                    decision = (goalie_stats.get("decision") or "").upper()
                    if decision == "W": row["W"] = 1
                    elif decision in ("OT", "OTL"): row["OTL"] = 1
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

def load_totals_data(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
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

def generate_report(roster, scoring, report_date_str, finished_games, boxscore_map, tomorrow_games_list):
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
    
    # 3. Update Totals (Using Fuzzy Matching)
    for norm, stats in new_stats.items():
        match_key = norm if norm in roster else None
        if not match_key:
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
    
    # 4. Build Output Data (Logic rewritten to use EXACT same matching as Totals)
    # Initialize all roster players with 0 stats
    player_entries = {}
    for key, r in roster.items():
        player_entries[key] = {
            "Player": r["Player"],
            "FantasyTeam": r["FantasyTeam"],
            "Stats": {k: 0 for k in ("G", "A", "PPP", "SOG", "HIT", "BLK", "W", "GA", "SV", "SO", "OTL")},
            "DailyPts": 0.0,
            "TotalPts": total_scores.get(key, 0.0)
        }

    # Overlay Daily Stats using the matching logic
    for norm, stats in daily_stats.items():
        match_key = norm if norm in roster else None
        if not match_key:
            last = norm.split()[-1] if norm.split() else ""
            cands = [k for k in roster.keys() if k.endswith(f" {last}") or k == last]
            if len(cands) == 1: match_key = cands[0]
        
        if match_key:
            # Found the roster match -> Update daily stats
            d_pts = compute_points(stats, scoring)
            player_entries[match_key]["Stats"] = stats
            player_entries[match_key]["DailyPts"] = d_pts

    final_player_list = list(player_entries.values())

    # Sort and Write Report
    write_text_report(final_player_list, finished_games, report_date_str, tomorrow_games_list)
    write_html_report(final_player_list, finished_games, report_date_str, tomorrow_games_list)
    push_to_github()

def write_text_report(players, games, report_date_str, upcoming):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = ["═══ CAMELOT CANIACS OLYMPIC UPDATE ═══"]
    lines.append(f"Daily Report for {report_date_str}")
    lines.append(f"Last Updated: {now_str}\n")
    
    lines.append("YESTERDAY'S GAMES:")
    if not games:
        lines.append(" (No completed games found for this date)")
    for g in games:
        lines.append(f" - {g.get('awayTeam',{}).get('abbrev')} @ {g.get('homeTeam',{}).get('abbrev')} (FINAL)")
    
    teams = defaultdict(list)
    for p in players: teams[p["FantasyTeam"]].append(p)
    
    # --- Top 5 Scorers (Daily) ---
    lines.append("\nTOP 5 STARS OF THE DAY:")
    active_today = [p for p in players if p['DailyPts'] > 0]
    all_sorted = sorted(active_today, key=lambda x: -x['DailyPts'])
    
    if not all_sorted:
         lines.append(" (No points scored today)")
    else:
        for i, p in enumerate(all_sorted[:5], 1):
            lines.append(f" {i}. {p['Player']} ({p['FantasyTeam']}): {p['DailyPts']:.1f}")
    
    # --- Team Daily Totals ---
    lines.append("\nTEAM DAILY PERFORMANCE:")
    team_daily_sums = sorted([(t, sum(x['DailyPts'] for x in m)) for t, m in teams.items()], key=lambda x: (-x[1], x[0]))
    
    for team, total in team_daily_sums:
        lines.append(f"{team} — Daily: {total:.1f}")
        lines.append("-" * 75)
        lines.append(f"{'Player':20} | {'Stats (Today)':30} | Daily | Total")
        
        for p in sorted(teams[team], key=lambda x: (-x['DailyPts'], -x['TotalPts'])):
            s = p['Stats']
            stat_str = ""
            
            if s.get('SV') or s.get('GA') or s.get('W') or s.get('OTL'):
                stat_str = f"W:{int(s['W'])} SV:{int(s['SV'])} GA:{int(s['GA'])}"
                if s.get('SO'): stat_str += f" SO:{int(s['SO'])}"
                if s.get('OTL'): stat_str += f" OTL:{int(s['OTL'])}"
            elif any(s.values()):
                stat_str = f"{int(s['G'])}G {int(s['A'])}A"
                if s.get('PPP'): stat_str += f" {int(s['PPP'])}PPP"
                stat_str += f" {int(s['SOG'])}S"
                if s.get('HIT'): stat_str += f" {int(s['HIT'])}H"
                if s.get('BLK'): stat_str += f" {int(s['BLK'])}B"
            else:
                stat_str = "-"

            lines.append(f"{p['Player'][:20]:20} | {stat_str:30} | {p['DailyPts']:5.1f} | {p['TotalPts']:5.1f}")
        lines.append("")
        
    lines.append("🏆 OVERALL STANDINGS (Total Tournament Points):")
    team_season_sums = defaultdict(float)
    for p in players:
        team_season_sums[p['FantasyTeam']] += p['TotalPts']
    
    sorted_standings = sorted(team_season_sums.items(), key=lambda x: -x[1])
    for rank, (team, pts) in enumerate(sorted_standings, 1):
        lines.append(f" {rank}. {team}: {pts:.1f}")
        
    lines.append("\nUPCOMING SCHEDULE (TODAY):")
    if not upcoming:
        lines.append(" (No games scheduled)")
    for g in upcoming:
        away = g.get('awayTeam', {}).get('abbrev') or "???"
        home = g.get('homeTeam', {}).get('abbrev') or "???"
        start = g.get('startTimeUTC', 'TBD')
        lines.append(f" - {away} vs {home} ({start})")
        
    with open(REPORT_TXT, "w") as f: f.write("\n".join(lines))
    print(f"Report written to {REPORT_TXT}")

def write_html_report(players, games, report_date_str, upcoming):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<html><head><style>
    body{{font-family:sans-serif; padding:20px; max-width:800px; margin:0 auto;}} 
    table{{border-collapse:collapse; width:100%}} 
    th,td{{border:1px solid #ddd; padding:8px; text-align:left}} th{{background-color:#f2f2f2}}
    .team{{background:#333; color:white; padding:10px; margin-top:20px}}
    .header{{margin-bottom:20px; border-bottom:2px solid #333; padding-bottom:10px;}}
    .standings{{background:#eef; padding:15px; margin-bottom:20px; border-radius:5px;}}
    .top5{{background:#fff3cd; padding:15px; margin-bottom:20px; border-radius:5px;}}
    </style></head><body>
    
    <div class="header">
        <h1>Camelot Caniacs Olympic Update</h1>
        <p><b>Report Date:</b> {report_date_str} | <b>Last Updated:</b> {now_str}</p>
    </div>
    """
    
    html += "<div class='top5'><h3>⭐ Top 5 Stars of the Day</h3><ul>"
    active_today = [p for p in players if p['DailyPts'] > 0]
    all_sorted = sorted(active_today, key=lambda x: -x['DailyPts'])
    if not all_sorted:
        html += "<li>No points scored today.</li>"
    else:
        for p in all_sorted[:5]:
            html += f"<li><b>{p['Player']}</b> ({p['FantasyTeam']}) - {p['DailyPts']:.1f} pts</li>"
    html += "</ul></div>"
    
    html += "<div class='standings'><h3>🏆 Overall Standings</h3><ol>"
    team_season_sums = defaultdict(float)
    for p in players: team_season_sums[p['FantasyTeam']] += p['TotalPts']
    sorted_standings = sorted(team_season_sums.items(), key=lambda x: -x[1])
    for team, pts in sorted_standings:
        html += f"<li><b>{team}</b>: {pts:.1f}</li>"
    html += "</ol></div>"

    teams = defaultdict(list)
    for p in players: teams[p["FantasyTeam"]].append(p)
    team_sums = sorted([(t, sum(x['DailyPts'] for x in m)) for t, m in teams.items()], key=lambda x: (-x[1], x[0]))
    
    html += "<h2>Daily Performance</h2>"
    for team, total in team_sums:
        html += f"<div class='team'><b>{team}</b> - Daily: {total:.1f}</div><table><tr><th>Player</th><th>Stats (Today)</th><th>Daily</th><th>Total</th></tr>"
        for p in sorted(teams[team], key=lambda x: (-x['DailyPts'], -x['TotalPts'])):
             s = p['Stats']
             stat_str = "-"
             if s.get('SV') or s.get('GA') or s.get('W') or s.get('OTL'):
                 stat_str = f"W:{int(s.get('W',0))} SV:{int(s.get('SV',0))} GA:{int(s.get('GA',0))}"
                 if s.get('SO', 0) > 0: stat_str += f" <b>SO:{int(s['SO'])}</b>"
                 if s.get('OTL', 0) > 0: stat_str += f" OTL:{int(s['OTL'])}"
             elif any(s.values()):
                 stat_str = f"{int(s.get('G',0))}G {int(s.get('A',0))}A {int(s.get('SOG',0))}S"
                 if s.get('PPP'): stat_str += f" {int(s.get('PPP'))}PPP"
                 if s.get('HIT'): stat_str += f" {int(s.get('HIT'))}H"
                 if s.get('BLK'): stat_str += f" {int(s.get('BLK'))}B"
             
             html += f"<tr><td>{p['Player']}</td><td>{stat_str}</td><td>{p['DailyPts']:.1f}</td><td>{p['TotalPts']:.1f}</td></tr>"
        html += "</table>"
    
    html += "<h3>📅 Upcoming Schedule (Today)</h3><ul>"
    if not upcoming: html += "<li>No games scheduled</li>"
    for g in upcoming:
        away = g.get('awayTeam', {}).get('abbrev') or "???"
        home = g.get('homeTeam', {}).get('abbrev') or "???"
        start = g.get('startTimeUTC', 'TBD')
        html += f"<li>{away} vs {home} ({start})</li>"
    html += "</ul></body></html>"

    with open(REPORT_HTML, "w") as f: f.write(html)

def push_to_github():
    import subprocess
    try:
        os.chdir(BASE_DIR)
        subprocess.run(["git", "add", "daily_report.txt", "daily_report.html", "totals.json"], check=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", f"Update: {datetime.now().strftime('%Y-%m-%d')}"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("🚀 GitHub updated successfully!")
    except Exception as e:
        print(f"⚠️ Git push failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", help="Date to generate report for (YYYY-MM-DD)")
    args = parser.parse_args()

    print("🏒 Camelot Caniacs - daily_stats running...")
    scoring = load_scoring()
    roster = load_rosters(ROSTERS_CSV)
    
    if args.date:
        report_date = args.date
    else:
        report_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    report_dt = datetime.strptime(report_date, "%Y-%m-%d")
    next_day = (report_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    
    games_report_day = fetch_schedule_for_date(report_date)
    games_next_day = fetch_schedule_for_date(next_day)
    
    finished_games = []
    finished_ids = []
    for g in games_report_day:
        state = (g.get("gameState") or g.get("status", {}).get("detailedState") or "").upper()
        if "FINAL" in state:
            finished_games.append(g)
            finished_ids.append(g.get("gamePk") or g.get("id"))

    if not finished_games:
        print(f"⚠️ No finished games found for {report_date}.")
        
    boxscore_map = {}
    for pid in finished_ids:
        boxscore_map[pid] = parse_boxscore_for_players(fetch_boxscore(pid))
        
    generate_report(roster, scoring, report_date, finished_games, boxscore_map, games_next_day)

if __name__ == "__main__":
    main()