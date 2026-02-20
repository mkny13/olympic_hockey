"""daily_stats.py

Robust daily boxscore aggregator for the Camelot Caniacs Olympic fantasy league.
Updates: 
- Fixes SOG (Shots on Goal) issue by mapping the correct API key ('sog').
- Implements 'git stash' workflow for robust Github syncing.
- Formats stat lines consistently in Text and HTML.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
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
        return {"G": 2, "A": 1, "PPP": 0.5, "SHP": 0.5, "GWG": 1, "HAT": 1, "SOG": 0.1, "HIT": 0.1, "BLK": 0.5, "W": 4, "GA": -2, "SV": 0.2, "SO": 3, "OTL": 1}

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

def fetch_play_by_play(game_pk: int) -> dict | None:
    data = api_get(f"gamecenter/{game_pk}/play-by-play")
    if data:
        return data
    # StatsAPI fallback shape (legacy)
    return api_get(f"game/{game_pk}/feed/live")

def _iter_goal_events(pbp: dict) -> List[dict]:
    if not pbp:
        return []
    # NHL API shape
    plays = pbp.get("plays")
    if isinstance(plays, list):
        return [e for e in plays if (e.get("typeDescKey") or "").lower() == "goal"]
    # StatsAPI shape
    all_plays = (((pbp.get("liveData") or {}).get("plays") or {}).get("allPlays")) or []
    goals = []
    for e in all_plays:
        if (((e.get("result") or {}).get("eventTypeId")) or "").upper() == "GOAL":
            goals.append(e)
    return goals

def _build_id_to_norm_from_box(box: dict) -> Dict[int, str]:
    out: Dict[int, str] = {}
    pbg = (box or {}).get("playerByGameStats") or {}
    for side in ("awayTeam", "homeTeam"):
        team = pbg.get(side, {}) or {}
        for role in ("forwards", "defense"):
            for p in team.get(role, []) or []:
                pid = p.get("playerId") or p.get("id")
                name = (p.get("name") or {}).get("default") or p.get("name")
                if pid and name:
                    out[int(pid)] = _norm(name)
    # StatsAPI fallback
    all_players = (((box or {}).get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
    for side in ("away", "home"):
        players = ((all_players.get(side) or {}).get("players")) or {}
        for pdata in players.values():
            person = pdata.get("person") or {}
            pid = person.get("id")
            name = person.get("fullName")
            if pid and name:
                out[int(pid)] = _norm(name)
    return out

def _extract_team_ids_from_box(box: dict) -> Dict[str, int]:
    out: Dict[str, int] = {}
    away = (box or {}).get("awayTeam") or {}
    home = (box or {}).get("homeTeam") or {}
    if away.get("id") is not None:
        out["away"] = int(away["id"])
    if home.get("id") is not None:
        out["home"] = int(home["id"])
    # StatsAPI fallback
    teams = (((box or {}).get("gameData") or {}).get("teams")) or {}
    if "away" not in out and ((teams.get("away") or {}).get("id") is not None):
        out["away"] = int((teams.get("away") or {}).get("id"))
    if "home" not in out and ((teams.get("home") or {}).get("id") is not None):
        out["home"] = int((teams.get("home") or {}).get("id"))
    return out

def _extract_final_score(box: dict, goal_events: List[dict]) -> Tuple[int, int]:
    away = ((box or {}).get("awayTeam") or {}).get("score")
    home = ((box or {}).get("homeTeam") or {}).get("score")
    if away is not None and home is not None:
        return int(away), int(home)
    # StatsAPI fallback line score
    ls_teams = (((box or {}).get("liveData") or {}).get("linescore") or {}).get("teams") or {}
    if ((ls_teams.get("away") or {}).get("goals") is not None) and ((ls_teams.get("home") or {}).get("goals") is not None):
        return int((ls_teams.get("away") or {}).get("goals")), int((ls_teams.get("home") or {}).get("goals"))
    # Last known event score as a final fallback
    away_score, home_score = 0, 0
    for e in goal_events:
        d = (e.get("details") or {}) if e.get("details") is not None else {}
        if d.get("awayScore") is not None and d.get("homeScore") is not None:
            away_score = int(d.get("awayScore"))
            home_score = int(d.get("homeScore"))
            continue
        about = e.get("about") or {}
        if about.get("goals"):
            g = about.get("goals") or {}
            if g.get("away") is not None and g.get("home") is not None:
                away_score = int(g.get("away"))
                home_score = int(g.get("home"))
    return away_score, home_score

def _event_strength_is_shorthanded(event: dict) -> bool:
    # Check common explicit fields first.
    details = event.get("details") or {}
    strength = details.get("strength") or details.get("strengthCode")
    if isinstance(strength, dict):
        strength = strength.get("code") or strength.get("name")
    sval = str(strength or "").upper().replace("-", "").replace(" ", "")
    if sval in {"SH", "SHG", "SHORTHANDED", "SHORTHANDEDGOAL"}:
        return True

    # StatsAPI uses result.strength.code/name on GOAL plays.
    result = event.get("result") or {}
    r_strength = result.get("strength") or {}
    r_code = str((r_strength.get("code") or "")).upper().replace("-", "").replace(" ", "")
    r_name = str((r_strength.get("name") or "")).upper().replace("-", "").replace(" ", "")
    if r_code in {"SH", "SHG"} or r_name in {"SHORTHANDED", "SHORTHANDEDGOAL"}:
        return True

    # Conservative text fallback for varying payloads.
    text = " ".join([
        str(details.get("eventDescription") or ""),
        str(result.get("description") or ""),
        str(event.get("typeDescKey") or ""),
    ]).lower()
    return "short" in text and "hand" in text

def _event_team_side(event: dict, team_ids: Dict[str, int]) -> str | None:
    details = event.get("details") or {}
    owner = details.get("eventOwnerTeamId") or ((event.get("team") or {}).get("id"))
    if owner is not None:
        owner = int(owner)
        if owner == team_ids.get("away"):
            return "away"
        if owner == team_ids.get("home"):
            return "home"
    return None

def _event_scorer_id(event: dict) -> int | None:
    details = event.get("details") or {}
    sid = details.get("scoringPlayerId")
    if sid is not None:
        return int(sid)
    players = event.get("players") or []
    for p in players:
        if ((p.get("playerType") or "").upper() in {"SCORER"}) and ((p.get("player") or {}).get("id") is not None):
            return int((p.get("player") or {}).get("id"))
    return None

def _event_assist_ids(event: dict) -> List[int]:
    details = event.get("details") or {}
    out = []
    for k in ("assist1PlayerId", "assist2PlayerId"):
        if details.get(k) is not None:
            out.append(int(details[k]))
    if out:
        return out
    players = event.get("players") or []
    for p in players:
        if ((p.get("playerType") or "").upper().startswith("ASSIST")) and ((p.get("player") or {}).get("id") is not None):
            out.append(int((p.get("player") or {}).get("id")))
    return out

def derive_bonus_stats_from_play_by_play(rows: List[dict], box: dict, pbp: dict) -> None:
    goal_events = _iter_goal_events(pbp)
    if not goal_events:
        return
    id_to_norm = _build_id_to_norm_from_box(box)
    if not id_to_norm:
        return

    by_norm = {_norm(r.get("Player") or ""): r for r in rows}
    shp_counts: Dict[str, int] = defaultdict(int)
    gwg_norm: str | None = None

    team_ids = _extract_team_ids_from_box(box)
    away_final, home_final = _extract_final_score(box, goal_events)
    winner_side = None
    loser_score = None
    if away_final > home_final:
        winner_side, loser_score = "away", home_final
    elif home_final > away_final:
        winner_side, loser_score = "home", away_final

    away_running, home_running = 0, 0
    for e in goal_events:
        details = e.get("details") or {}
        if details.get("awayScore") is not None and details.get("homeScore") is not None:
            away_running = int(details.get("awayScore"))
            home_running = int(details.get("homeScore"))
        else:
            side = _event_team_side(e, team_ids)
            if side == "away":
                away_running += 1
            elif side == "home":
                home_running += 1

        scorer_id = _event_scorer_id(e)
        if _event_strength_is_shorthanded(e):
            for pid in [scorer_id] + _event_assist_ids(e):
                if pid is None:
                    continue
                norm = id_to_norm.get(int(pid))
                if norm:
                    shp_counts[norm] += 1

        if winner_side and loser_score is not None and gwg_norm is None:
            winning_score = away_running if winner_side == "away" else home_running
            if winning_score >= (loser_score + 1):
                if scorer_id is not None and id_to_norm.get(int(scorer_id)):
                    gwg_norm = id_to_norm[int(scorer_id)]

    for norm, v in shp_counts.items():
        if norm in by_norm:
            by_norm[norm]["SHP"] = max(int(by_norm[norm].get("SHP", 0)), v)
    if gwg_norm and gwg_norm in by_norm:
        by_norm[gwg_norm]["GWG"] = max(int(by_norm[gwg_norm].get("GWG", 0)), 1)

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
                    row = {"Player": name, "G":0, "A":0, "PPP":0, "SHP":0, "GWG":0, "HAT":0, "SOG":0, "HIT":0, "BLK":0, "W":0, "GA":0, "SV":0, "SO":0, "OTL":0}
                    row["G"] = p.get("goals", 0)
                    row["A"] = p.get("assists", 0)
                    row["PPP"] = p.get("powerPlayGoals", 0) + p.get("powerPlayAssists", 0)
                    row["GWG"] = p.get("gameWinningGoals", p.get("gameWinningGoal", 0))
                    row["SHP"] = p.get("shortHandedGoals", p.get("shorthandedGoals", 0)) + p.get("shortHandedAssists", p.get("shorthandedAssists", 0))
                    row["HAT"] = 1 if int(row["G"]) >= 3 else 0
                    
                    # FIX: NHL Olympic feed uses 'sog', regular season often uses 'shots'
                    row["SOG"] = p.get("sog", p.get("shots", 0))
                    
                    # Note: These keys exist but are 0 in Olympic feed
                    row["HIT"] = p.get("hits", 0)
                    row["BLK"] = p.get("blockedShots", 0)
                    rows.append(row)
            for g in team.get("goalies", []) or []:
                name = (g.get("name") or {}).get("default") or g.get("name")
                row = {"Player": name, "G":0, "A":0, "PPP":0, "SHP":0, "GWG":0, "HAT":0, "SOG":0, "HIT":0, "BLK":0, "W":0, "GA":0, "SV":0, "SO":0, "OTL":0}
                row["GA"] = g.get("goalsAgainst", 0)
                row["SV"] = g.get("saves", 0)
                decision = (g.get("decision") or "").upper()
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
            agg[key] = {k: 0 for k in ("G", "A", "PPP", "SHP", "GWG", "HAT", "SOG", "HIT", "BLK", "W", "GA", "SV", "SO", "OTL")}
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

def generate_report(roster, scoring, report_date_str, finished_games, boxscore_map, tomorrow_games_list, do_push=True):
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
        player_entries[key] = {"Player": r["Player"], "FantasyTeam": r["FantasyTeam"], "Stats": {k: 0 for k in ("G", "A", "PPP", "SHP", "GWG", "HAT", "SOG", "HIT", "BLK", "W", "GA", "SV", "SO", "OTL")}, "DailyPts": 0.0, "TotalPts": total_scores.get(key, 0.0)}

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
    if do_push:
        push_to_github()

def build_boxscore_map_for_games(games: List[dict]) -> Dict[int, List[dict]]:
    box_map: Dict[int, List[dict]] = {}
    for g in games:
        game_pk = g.get("gamePk") or g.get("id")
        if not game_pk:
            continue
        box = fetch_boxscore(game_pk)
        rows = parse_boxscore_for_players(box)
        if rows:
            pbp = fetch_play_by_play(game_pk)
            derive_bonus_stats_from_play_by_play(rows, box, pbp)
        box_map[game_pk] = rows
    return box_map

def format_game_time(utc_str: str) -> str:
    if not utc_str: return "TBD"
    try:
        dt_ny = datetime.fromisoformat(utc_str.replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York"))
        return dt_ny.strftime('%I:%M %p ET')
    except Exception: return utc_str

def format_stat_line(s: dict) -> str:
    """Returns a formatted string for stats, including bonus skater stats."""
    if s.get('SV') or s.get('W') or s.get('GA'):
        return f"W:{int(s['W'])} SV:{int(s['SV'])} GA:{int(s['GA'])}"
    
    if not any(s.values()):
        return "-"
        
    base = f"{int(s['G'])}G {int(s['A'])}A {int(s['SOG'])}S"
    extras = []
    if s.get('PPP'): extras.append(f"{int(s['PPP'])}PPP")
    if s.get('SHP'): extras.append(f"{int(s['SHP'])}SHP")
    if s.get('GWG'): extras.append(f"{int(s['GWG'])}GWG")
    if s.get('HAT'): extras.append(f"{int(s['HAT'])}HAT")
    if s.get('HIT'): extras.append(f"{int(s['HIT'])}H")
    if s.get('BLK'): extras.append(f"{int(s['BLK'])}B")
    
    if extras:
        return f"{base} {' '.join(extras)}"
    return base

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
            stat = format_stat_line(p['Stats'])
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
            stat = format_stat_line(p['Stats'])
            html += f"<tr><td>{p['Player']}</td><td>{stat}</td><td class='num'>{p['DailyPts']:.1f}</td><td class='num'>{p['TotalPts']:.1f}</td></tr>"
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
    parser.add_argument("--no-push", action="store_true", help="Generate reports/totals but skip git push.")
    parser.add_argument("--rebuild-from", help="Rebuild totals from this start date (YYYY-MM-DD), inclusive.")
    parser.add_argument("--rebuild-to", help="Rebuild totals through this end date (YYYY-MM-DD), inclusive.")
    args = parser.parse_args()
    scoring, roster = load_scoring(), load_rosters(ROSTERS_CSV)
    rep_date = args.date or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    do_push = not args.no_push

    if args.rebuild_from:
        start_dt = datetime.strptime(args.rebuild_from, "%Y-%m-%d")
        end_dt = datetime.strptime(args.rebuild_to or rep_date, "%Y-%m-%d")
        if end_dt < start_dt:
            raise ValueError("--rebuild-to must be on/after --rebuild-from")
        save_totals_data(TOTALS_JSON, {"scores": {}, "processed_games": []})

        current = start_dt
        while current <= end_dt:
            day = current.strftime("%Y-%m-%d")
            tomorrow = (current + timedelta(days=1)).strftime("%Y-%m-%d")
            games_rep = fetch_schedule_for_date(day)
            games_next = fetch_schedule_for_date(tomorrow)
            finished = [g for g in games_rep if "FINAL" in (g.get("gameState") or g.get("status", {}).get("detailedState") or "").upper()]
            box_map = build_boxscore_map_for_games(finished)
            is_final_day = (current == end_dt)
            generate_report(roster, scoring, day, finished, box_map, games_next, do_push=(do_push and is_final_day))
            current += timedelta(days=1)
        return

    rep_dt = datetime.strptime(rep_date, "%Y-%m-%d")
    games_rep = fetch_schedule_for_date(rep_date)
    games_next = fetch_schedule_for_date((rep_dt + timedelta(days=1)).strftime("%Y-%m-%d"))
    finished = [g for g in games_rep if "FINAL" in (g.get("gameState") or g.get("status", {}).get("detailedState") or "").upper()]
    box_map = build_boxscore_map_for_games(finished)
    generate_report(roster, scoring, rep_date, finished, box_map, games_next, do_push=do_push)

if __name__ == "__main__":
    main()
