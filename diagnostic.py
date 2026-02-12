import requests
import pandas as pd

# Try the two most likely 2026 endpoints
URLS = [
    ("Summary (Game Type 9)", "https://api.nhle.com/stats/rest/en/skater/summary?cayenneExp=seasonId=20252026%20and%20gameTypeId=9"),
    ("Realtime (Game Type 9)", "https://api.nhle.com/stats/rest/en/skater/realtime?cayenneExp=seasonId=20252026%20and%20gameTypeId=9")
]

print("🔍 SCANNING FOR 2026 DATA...\n")

for label, url in URLS:
    print(f"--- Checking: {label} ---")
    try:
        data = requests.get(url, timeout=10).json()
        if 'data' in data and data['data']:
            df = pd.DataFrame(data['data'])
            print(f"✅ FOUND DATA! ({len(df)} records)")
            
            # Print the first 5 names to see the format
            if 'skaterFullName' in df.columns:
                print("Names found:", df['skaterFullName'].head(5).tolist())
            elif 'player' in df.columns:
                print("Names found:", df['player'].head(5).tolist())
            else:
                print("Columns found:", df.columns.tolist())
        else:
            print("❌ Response received, but 'data' list is empty.")
    except Exception as e:
        print(f"❌ Error: {e}")
    print("\n")
