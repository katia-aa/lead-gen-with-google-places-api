"""Lead generation using Google Places API and OpenAI GPT-4."""

import os
import logging
import requests
import openai
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

GOOGLE_PLACES_TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
GOOGLE_PLACES_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"


def search_google_places(query: str, location: str, count: int) -> List[Dict]:
    """Search Google Places and return a list of place dictionaries."""
    params = {
        "query": f"{query} in {location}",
        "key": GOOGLE_API_KEY,
    }
    results = []
    next_page_token: Optional[str] = None

    while len(results) < count:
        if next_page_token:
            params["pagetoken"] = next_page_token
        response = requests.get(GOOGLE_PLACES_TEXT_SEARCH_URL, params=params, timeout=10)
        if response.status_code != 200:
            logging.error("Google Places text search failed: %s", response.text)
            break
        data = response.json()
        results.extend(data.get("results", []))
        next_page_token = data.get("next_page_token")
        if not next_page_token:
            break
        logging.info("Waiting for next page token...")
        import time
        time.sleep(2)
    return results[:count]


def get_place_details(place_id: str) -> Dict:
    """Retrieve details for a place using its place_id."""
    params = {
        "place_id": place_id,
        "fields": "name,formatted_phone_number,website,types",
        "key": GOOGLE_API_KEY,
    }
    response = requests.get(GOOGLE_PLACES_DETAILS_URL, params=params, timeout=10)
    if response.status_code != 200:
        logging.error("Google Places details request failed: %s", response.text)
        return {}
    return response.json().get("result", {})


def extract_contact_info_from_website(url: str) -> Dict[str, Optional[str]]:
    """Fetch a webpage and use GPT-4 to extract contact info."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logging.error("Failed to fetch %s: %s", url, e)
        return {"email": None, "instagram": None}

    prompt = (
        "Extract a single email address if present, and a single Instagram URL if present, from the following HTML. "
        "Respond in JSON with keys 'email' and 'instagram'. If not found, use null. HTML:" + resp.text
    )
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = completion.choices[0].message["content"].strip()
        import json
        data = json.loads(content)
        return {"email": data.get("email"), "instagram": data.get("instagram")}
    except Exception as e:
        logging.error("OpenAI extraction failed: %s", e)
        return {"email": None, "instagram": None}


def get_leads(query: str, location: str, count: int) -> List[Dict]:
    """Return a list of leads with contact information."""
    leads: List[Dict] = []
    search_results = search_google_places(query, location, count)
    logging.info("Found %d places", len(search_results))

    for result in search_results:
        place_id = result.get("place_id")
        details = get_place_details(place_id)
        name = details.get("name")
        phone = details.get("formatted_phone_number")
        website = details.get("website")
        category = details.get("types", [None])[0] if details.get("types") else None

        email = None
        instagram = None
        if website:
            contact_info = extract_contact_info_from_website(website)
            email = contact_info.get("email")
            instagram = contact_info.get("instagram")
        elif phone:
            logging.info("No website for %s; using phone number", name)

        lead = {
            "name": name,
            "category": category,
            "phone": phone,
            "website": website,
            "email": email,
            "instagram": instagram,
        }
        leads.append(lead)
    return leads


if __name__ == "__main__":
    query = "wedding venues"
    location = "Toronto"
    count = 5
    leads = get_leads(query, location, count)
    for lead in leads:
        print(lead)

