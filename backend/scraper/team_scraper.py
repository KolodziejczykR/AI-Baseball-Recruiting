from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup, Tag
import re
import os
from dotenv import load_dotenv

# TODO: fix scraper as it is not working
# TODO: implement more tests
# TODO: look at recruiting class scraper

load_dotenv()  # Loads variables from .env

router = APIRouter()

# --- Pydantic models for request/response ---
class TeamInfoRequest(BaseModel):
    team: str
    position: str
    class_year: int

class PlayerStats(BaseModel):
    name: str
    AB: int
    AVG: float
    OBP: float
    SLG: float
    FLD_percent: float

# --- SerpAPI integration ---
SERPAPI_KEY = os.getenv('SERPAPI_KEY')  # Set this environment variable with your SerpAPI key

# --- SerpAPI search helper ---
def search_roster_url(team: str) -> str:
    if not SERPAPI_KEY:
        raise HTTPException(status_code=500, detail="SerpAPI key not set. Please set the SERPAPI_KEY environment variable.")
    query = f"{team} baseball roster"
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 5,
        "engine": "google"
    }
    serp_api_url = "https://serpapi.com/search"
    resp = requests.get(serp_api_url, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Failed to fetch search results from SerpAPI")
    data = resp.json()
    # Try to find the first organic result
    for result in data.get("organic_results", []):
        link = result.get("link")
        if link and "roster" in link:
            return link
    # Fallback: return first organic result
    if data.get("organic_results"):
        return data["organic_results"][0].get("link")
    raise HTTPException(status_code=404, detail="No roster link found in search results.")

def parse_roster_table(table, position_filter):
    # Get header mapping
    header_row = table.find('tr')
    headers = [th.get_text(strip=True).upper() for th in header_row.find_all(['th', 'td'])]
    col_map = {name: idx for idx, name in enumerate(headers)}
    players = []
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if not cols or len(cols) < len(headers):
            continue
        position = cols[col_map.get('POSITION')].get_text(strip=True) if 'POSITION' in col_map else ''
        if position_filter and position.upper() != position_filter.upper():
            continue
        name = cols[col_map.get('NAME')].get_text(strip=True) if 'NAME' in col_map else ''
        # Example: add more fields as needed, using col_map
        player = {
            'name': name,
            'position': position,
        }
        # Optionally extract stats if present
        for stat in ['AB', 'AVG', 'OBP', 'SLG', 'FLD%']:
            idx = col_map.get(stat)
            if idx is not None and idx < len(cols):
                val = cols[idx].get_text(strip=True)
                try:
                    player[stat] = float(val) if '.' in val else int(val)
                except Exception:
                    player[stat] = val
        players.append(player)
    return players

# --- Real scraping functions ---
def scrape_current_roster(team: str, position: str) -> Dict[str, Any]:
    url = search_roster_url(team)
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Failed to fetch roster page")
    soup = BeautifulSoup(response.text, 'html.parser')

    # Try to find a table with id 'players-table', else fallback to first table
    table = soup.find('table', id='players-table') or soup.find('table')
    if not isinstance(table, Tag):
        raise HTTPException(status_code=404, detail="Roster table not found (not a valid table tag)")

    players = parse_roster_table(table, position)
    return {
        "count": len(players),
        "players": players
    }

def scrape_recruiting_class(team: str, position: str, class_year: int) -> Dict[str, Any]:
    # Placeholder: Construct the URL for the team's PBR recruiting class page
    url = f"https://www.prepbaseballreport.com/college-commitments/{team.replace(' ', '-').lower()}?class={class_year}"
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Failed to fetch recruiting class page")
    soup = BeautifulSoup(response.text, 'html.parser')

    # Placeholder: Find the commitments table (update selector as needed)
    table = soup.find('table', {'id': 'commitments-table'})
    if not isinstance(table, Tag):
        raise HTTPException(status_code=404, detail="Commitments table not found (not a valid table tag)")

    count = 0
    for row in table.find_all('tr')[1:]:
        if not isinstance(row, Tag):
            continue
        cols = row.find_all('td')
        if not cols or len(cols) < 4:
            continue
        recruit_position = cols[2].get_text(strip=True)
        recruit_class = cols[3].get_text(strip=True)
        if recruit_position.upper() == position.upper() and str(class_year) in recruit_class:
            count += 1
    return {
        "year": class_year,
        "position": position,
        "count": count
    }

# --- API Route ---
@router.post("/scrape/team-info")
def get_team_info(request: TeamInfoRequest):
    current_roster = scrape_current_roster(request.team, request.position)
    recruiting_class = scrape_recruiting_class(request.team, request.position, request.class_year)
    return {
        "current_roster": current_roster,
        "recruiting_class": recruiting_class
    } 