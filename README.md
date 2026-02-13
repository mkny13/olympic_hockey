# Camelot Caniacs Olympic Tracker

A little toy for my fantasy hockey league to pretend the olympics are an in season tournament for us. This was whipped together using Built with Gemini Pro and VS Code agent using GPT-5 Mini..

## How It Works

This script acts as a daily "boxscore scraper" for the NHL/Olympic API. It compares game stats against our league's specific rosters and scoring settings to generate updates.

- **The Brains:** `daily_stats.py` handles the API calls, fuzzy-matches player names (because special characters are the enemy), and manages the math.
- **Persistence:** It tracks tournament-long scores in `totals.json` so we don't lose progress between updates.
- **Automation:** It’s designed to run, generate reports, and then push the updated results directly back to this repo.

## Scoring Settings

We use a custom scoring weight defined in `scoring_config.py`:

| Skaters | Points | | Goalies | Points |
| :--- | :--- | :--- | :--- | :--- |
| **G** | 2.0 | | **W** | 4.0 |
| **A** | 1.0 | | **GA** | -2.0 |
| **PPP** | 0.5 | | **SV** | 0.2 |
| **SOG/HIT** | 0.1 | | **SO** | 3.0 |
| **BLK** | 0.5 | | **OTL** | 1.0 |

## Usage

If you're running this locally:

1.  **Prep the Roster:** Ensure `rosters.csv` is updated with the current `Player`, `FantasyTeam`, and `OlympicTeam`. (I gave Gemini raw inputs that is transformed for this.)
2.  **Install Deps:** `pip install requests`
3.  **Run it:** ```bash
    # Run for yesterday's games (default)
    python daily_stats.py
    
    # Run for a specific date
    python daily_stats.py --date 2026-02-15
    ```

## Outputs

- **`daily_report.txt`:** A clean, mono-spaced summary for quick reading.
- **`daily_report.html`:** A slightly prettier version with standings and the "Top 5 Stars of the Day."
- **`totals.json`:** The source of truth for the overall leaderboard.

---
*Built with Gemini Pro and VS Code agent using GPT-5 Mini.*
