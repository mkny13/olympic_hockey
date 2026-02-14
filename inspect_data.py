import requests
import json
import sys

# 1. Config
DATE = "2026-02-13" # The date of the games in question
API_BASE = "https://api-web.nhle.com/v1"

def check_data():
    print(f"🔍 Inspecting NHL API data for {DATE}...")
    
    # 2. Get Schedule to find a Game ID
    sched_url = f"{API_BASE}/schedule/{DATE}"
    try:
        r = requests.get(sched_url)
        data = r.json()
    except Exception as e:
        print(f"❌ Failed to fetch schedule: {e}")
        return

    game_id = None
    game_str = ""
    
    # Logic to find the Canada/Switzerland game (or any game)
    days = data.get("gameWeek", [])
    for day in days:
        if day.get("date") == DATE:
            for g in day.get("games", []):
                # Grab the first finished game we find
                if "FINAL" in (g.get("gameState") or "").upper():
                    game_id = g.get("id")
                    away = g.get("awayTeam", {}).get("abbrev")
                    home = g.get("homeTeam", {}).get("abbrev")
                    game_str = f"{away} vs {home}"
                    break
    
    if not game_id:
        print("❌ No final games found for this date to inspect.")
        return

    print(f"✅ Found Game: {game_str} (ID: {game_id})")

    # 3. Get Boxscore
    box_url = f"{API_BASE}/gamecenter/{game_id}/boxscore"
    print(f"📥 Fetching boxscore from: {box_url}")
    
    try:
        box = requests.get(box_url).json()
    except Exception as e:
        print(f"❌ Failed to fetch boxscore: {e}")
        return

    # 4. Extract one Forward to see their keys
    pbg = box.get("playerByGameStats", {})
    home_team = pbg.get("homeTeam", {})
    forwards = home_team.get("forwards", [])

    if not forwards:
        print("❌ No forward stats found in boxscore.")
    else:
        # Grab a player who definitely had stats (like a goal scorer if possible, otherwise just the first one)
        player = forwards[0] 
        name = (player.get("name") or {}).get("default")
        
        print("\n" + "="*40)
        print(f" RAW DATA FOR: {name}")
        print("="*40)
        print(json.dumps(player, indent=2))
        print("="*40)
        
        # 5. Check specific missing keys
        print("\n🔎 KEY CHECK:")
        for key in ["goals", "assists", "shots", "hits", "blockedShots", "powerPlayGoals"]:
            val = player.get(key, "MISSING")
            print(f" - {key}: {val}")

if __name__ == "__main__":
    check_data()