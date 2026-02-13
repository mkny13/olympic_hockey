"""daily_stats.py

Robust daily boxscore aggregator for the Camelot Caniacs Olympic fantasy league.
Updates: 
- Implements 'git stash' workflow to fix "unstaged changes" error during rebase.
- Ensures reports are pushed even if the script file itself is modified locally.
- changed column header from "Stats (Today)" to "Stats for {Month Day}"
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
from zoneinfo import ZoneInfo

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
            for role in ("forwards", "defense"):
                for p in team.get(role, []) or []:
                    name = (p.get("name") or {}).get("default") or p.get("name")
                    row = {"Player": name, "G":0, "A":0, "PPP":0, "SOG":0, "HIT":0, "BLK":0, "W":0, "GA":0, "SV":0, "SO":0, "OTL":0}
                    row["G"], row["A"] = p.get("goals", 0), p.get("assists", 0)
                    row["PPP"] = p.get("powerPlayGoals", 0) + p.get("powerPlayAssists", 0)
                    row["SOG"], row["HIT"], row["BLK"] = p.get("shots", 0), p.get("hits", 0), p.get("blockedShots", 0)
                    rows.append(row)
            for g in team.get("goalies", []) or []:
                name = (g.get("name") or {}).get("default") or g.get("name")
                row = {"Player": name, "G":0, "A":0, "PPP":0, "SOG":0, "HIT":0, "BLK":0, "W":0, "GA":0, "SV":0, "SO":0, "OTL":0}
                row["GA"], row["SV"] = g.get("goalsAgainst", 0), g.get("saves", 0)
                dec = (g.get("decision") or "").upper()
                if dec == "W": row["W"] = 1
                elif dec in ("OT", "OTL"): row["OTL"] = 1
                if row["GA"] == 0 and row["W"] == 1 and row["SV"] > 0: row["SO"] = 1
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
            if k != "Player": agg[key][k] += int(r.get(k, 0))
    return agg

def load_totals_data(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return data if isinstance(data.get("scores"), dict) else {"scores": data, "processed_games": []}
        except Exception: pass
    return {"scores": {}, "processed_games": []}

def save_totals_data(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

def compute_points(stats: dict, scoring: dict) -> float:
    return sum(float(stats.get(k, 0)) * float(scoring.get(k, 0)) for k in scoring)

def generate_report(roster, scoring, report_date_str, finished_games, boxscore_map, tomorrow_games_list):
    data = load_totals_data(TOTALS_JSON)
    total_scores = data.get("scores", {})
    processed_ids = set(data.get("processed_games", []))
    
    all_window_rows = []
    for pk, rows in boxscore_map.items(): all_window_rows.extend(rows)
    daily_stats = aggregate_by_player(all_window_rows)
    
    new_rows, new_game_ids = [], []
    for pk, rows in boxscore_map.items():
        if pk not in processed_ids:
            new_rows.extend(rows), new_game_ids.append(pk)
    
    for norm, stats in aggregate_by_player(new_rows).items():
        match_key = norm if norm in roster else None
        if not match_key:
            last = norm.split()[-1] if norm.split() else ""
            cands = [k for k in roster.keys() if k.endswith(f" {last}") or k == last]
            if len(cands) == 1: match_key = cands[0]
        if match_key:
            total_scores[match_key] = total_scores.get(match_key, 0.0) + compute_points(stats, scoring)

    data["scores"], data["processed_games"] = total_scores, list(processed_ids.union(set(new_game_ids)))
    save_totals_data(TOTALS_JSON, data)
    
    player_entries = {}
    for key, r in roster.items():
        player_entries[key] = {"Player": r["Player"], "FantasyTeam": r["FantasyTeam"], "Stats": {k: 0 for k in ("G", "A", "PPP", "SOG", "HIT", "BLK", "W", "GA", "SV", "SO", "OTL")}, "DailyPts": 0.0, "TotalPts": total_scores.get(key, 0.0)}

    for norm, stats in daily_stats.items():
        match_key = norm if norm in roster else None
        if not match_key:
            last = norm.split()[-1] if norm.split() else ""
            cands = [k for k in roster.keys() if k.endswith(f" {last}") or k == last]
            if len(cands) == 1: match_key = cands[0]
        if match_key:
            player_entries[match_key]["Stats"] = stats
            player_entries[match_key]["DailyPts"] = compute_points(stats, scoring)

    write_text_report(list(player_entries.values()), finished_games, report_date_str, tomorrow_games_list)
    write_html_report(list(player_entries.values()), finished_games, report_date_str, tomorrow_games_list)
    push_to_github()

def format_game_time(utc_str: str) -> str:
    if not utc_str: return "TBD"
    try:
        dt_ny = datetime.fromisoformat(utc_str.replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York"))
        return dt_ny.strftime('%I:%M %p ET')
    except Exception: return utc_str

def write_text_report(players, games, report_date_str, upcoming):
    now_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M %p ET")
    short_date = datetime.strptime(report_date_str, "%Y-%m-%d").strftime("%b %d")
    header_col = f"Stats for {short_date}"
    
    lines = ["═══ CAMELOT CANIACS OLYMPIC UPDATE ═══", f"Daily Report for {report_date_str}", f"Last Updated: {now_str}\n", "YESTERDAY'S GAMES:"]
    if not games: lines.append(" (No completed games found)")
    for g in games:
        away, home = g.get('awayTeam',{}).get('abbrev', "???"), g.get('homeTeam',{}).get('abbrev', "???")
        lines.append(f" - {away} ({g.get('awayTeam',{}).get('score',0)}) @ {home} ({g.get('homeTeam',{}).get('score',0)}) (FINAL)")
    
    stars = sorted([p for p in players if p['DailyPts'] > 0], key=lambda x: -x['DailyPts'])
    lines.extend(["\nTOP 5 STARS OF THE DAY:"] + ([f" {i}. {p['Player']} ({p['FantasyTeam']}): {p['DailyPts']:.1f}" for i, p in enumerate(stars[:5], 1)] if stars else [" (No points today)"]))
    
    teams = defaultdict(list)
    for p in players: teams[p["FantasyTeam"]].append(p)
    for team, total in sorted([(t, sum(x['DailyPts'] for x in m)) for t, m in teams.items()], key=lambda x: (-x[1], x[0])):
        lines.extend([f"\n{team} — Daily: {total:.1f}", "-" * 75, f"{'Player':20} | {header_col:30} | Daily | Total"])
        for p in sorted(teams[team], key=lambda x: (-x['DailyPts'], -x['TotalPts'])):
            s = p['Stats']
            stat = f"W:{int(s['W'])} SV:{int(s['SV'])} GA:{int(s['GA'])}" if s.get('SV') or s.get('W') else f"{int(s['G'])}G {int(s['A'])}A {int(s['SOG'])}S" if any(s.values()) else "-"
            lines.append(f"{p['Player'][:20]:20} | {stat:30} | {p['DailyPts']:5.1f} | {p['TotalPts']:5.1f}")
            
    standings = sorted([(t, sum(p['TotalPts'] for p in m)) for t, m in teams.items()], key=lambda x: -x[1])
    lines.extend(["\n🏆 OVERALL STANDINGS:",] + [f" {i}. {t}: {pts:.1f}" for i, (t, pts) in enumerate(standings, 1)])
    lines.extend(["\nUPCOMING:",] + [f" - {g.get('awayTeam',{}).get('abbrev')} vs {g.get('homeTeam',{}).get('abbrev')} ({format_game_time(g.get('startTimeUTC'))})" for g in upcoming])
    with open(REPORT_TXT, "w") as f: f.write("\n".join(lines))

def write_html_report(players, games, report_date_str, upcoming):
    now_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M %p ET")
    short_date = datetime.strptime(report_date_str, "%Y-%m-%d").strftime("%b %d")
    
    html = f"""<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'><title>Caniacs Update</title><style>:root{{--primary:#1a1a1a;--accent:#d32f2f;--bg:#f9f9f9;--card:#fff;--border:#e0e0e0}}body{{font-family:-apple-system,sans-serif;background:var(--bg);color:var(--primary);margin:0;padding:20px}}.container{{max-width:800px;margin:0 auto}}header{{border-bottom:3px solid var(--primary);padding-bottom:15px;margin-bottom:25px}}h1{{margin:0;font-size:1.8rem}}.meta{{color:#666;font-size:.9rem}}.card{{background:var(--card);border-radius:8px;padding:15px;margin-bottom:20px;border:1px solid var(--border);box-shadow:0 2px 4px rgba(0,0,0,0.05)}}table{{width:100%;border-collapse:collapse;font-size:.95rem}}th,td{{padding:10px;border-bottom:1px solid var(--border);text-align:left}}th{{background:#f4f4f4}}.num{{text-align:right}}.team-h{{background:var(--primary);color:#fff;padding:10px 15px;border-radius:6px 6px 0 0;display:flex;justify-content:space-between;margin-top:25px}}.game-s{{font-weight:700}}</style></head><body><div class='container'><header><h1>Camelot Caniacs Olympic Update</h1><div class='meta'>Date: {report_date_str} &bull; Updated: {now_str}</div></header><section class='card'><h3>📅 Yesterday's Games</h3><ul>"""
    for g in (games or []): html += f"<li>{g.get('awayTeam',{}).get('abbrev')} <span class='game-s'>({g.get('awayTeam',{}).get('score',0)})</span> @ {g.get('homeTeam',{}).get('abbrev')} <span class='game-s'>({g.get('homeTeam',{}).get('score',0)})</span></li>"
    html += "</ul></section><section class='card' style='background:#fffde7'><h3>⭐ Top 5 Stars of the Day</h3><ol>"
    for p in sorted([p for p in players if p['DailyPts'] > 0], key=lambda x: -x['DailyPts'])[:5]: html += f"<li><b>{p['Player']}</b> ({p['FantasyTeam']}) — {p['DailyPts']:.1f} pts</li>"
    html += "</ol></section><h2>Team Daily Performance</h2>"
    teams = defaultdict(list)
    for p in players: teams[p["FantasyTeam"]].append(p)
    for team, total in sorted([(t, sum(x['DailyPts'] for x in m)) for t, m in teams.items()], key=lambda x: (-x[1], x[0])):
        html += f"<div class='team-h'><span>{team}</span><span>Daily: {total:.1f}</span></div><div class='card' style='border-radius:0 0 6px 6px;border-top:none;margin-bottom:20px'><table><tr><th>Player</th><th>Stats for {short_date}</th><th class='num'>Daily</th><th class='num'>Total</th></tr>"
        for p in sorted(teams[team], key=lambda x: (-x['DailyPts'], -x['TotalPts'])):
            s, st = p['Stats'], "-"
            if s.get('SV') or s.get('W'): st = f"W:{int(s['W'])} SV:{int(s['SV'])} GA:{int(s['GA'])}"
            elif any(s.values()): st = f"{int(s['G'])}G {int(s['A'])}A {int(s['SOG'])}S"
            html += f"<tr><td>{p['Player']}</td><td>{st}</td><td class='num'>{p['DailyPts']:.1f}</td><td class='num'>{p['TotalPts']:.1f}</td></tr>"
        html += "</table></div>"
    html += "<section class='card'><h3>🏆 Overall Standings</h3><ol>"
    for team, pts in sorted([(t, sum(p['TotalPts'] for p in m)) for t, m in teams.items()], key=lambda x: -x[1]): html += f"<li><b>{team}</b>: {pts:.1f}</li>"
    html += "</ol></section><section class='card'><h3>📅 Upcoming Schedule</h3><ul>"
    for g in (upcoming or []): html += f"<li>{g.get('awayTeam',{}).get('abbrev')} vs {g.get('homeTeam',{}).get('abbrev')} ({format_game_time(g.get('startTimeUTC'))})</li>"
    html += "</ul></section></div></body></html>"
    with open(REPORT_HTML, "w") as f: f.write(html)

def push_to_github():
    import subprocess
    try:
        os.chdir(BASE_DIR)
        
        # 1. Commit first to "stage" our locally generated reports
        subprocess.run(["git", "add", "daily_report.txt", "daily_report.html", "totals.json"], check=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", f"Update: {datetime.now().strftime('%Y-%m-%d')}"], check=True)
        
        # 2. Stash unstaged changes (like this script file if modified)
        subprocess.run(["git", "stash"], check=False)
        
        # 3. Pull and Rebase (now safe because directory is clean)
        subprocess.run(["git", "pull", "origin", "main", "--rebase", "-X", "theirs"], check=True)
        
        # 4. Push
        subprocess.run(["git", "push", "origin", "main"], check=True)
        
        # 5. Restore stashed changes
        subprocess.run(["git", "stash", "pop"], check=False)
        
        print("🚀 GitHub updated successfully!")
    except Exception as e: print(f"⚠️ Git push failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", help="Date (YYYY-MM-DD)")
    args = parser.parse_args()
    scoring, roster = load_scoring(), load_rosters(ROSTERS_CSV)
    rep_date = args.date or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    rep_dt = datetime.strptime(rep_date, "%Y-%m-%d")
    games_rep, games_next = fetch_schedule_for_date(rep_date), fetch_schedule_for_date((rep_dt + timedelta(days=1)).strftime("%Y-%m-%d"))
    finished = [g for g in games_rep if "FINAL" in (g.get("gameState") or g.get("status", {}).get("detailedState") or "").upper()]
    box_map = { (g.get("gamePk") or g.get("id")): parse_boxscore_for_players(fetch_boxscore(g.get("gamePk") or g.get("id"))) for g in finished }
    generate_report(roster, scoring, rep_date, finished, box_map, games_next)

if __name__ == "__main__":
    main()