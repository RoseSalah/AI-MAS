import os
import requests
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("SERP_API_KEY")

# Define the query parameters
params = {
    "engine": "google",        # Search engine (e.g., google, bing, etc.)
    "q": "market analysis tools",  # Query string
    "api_key": API_KEY         # Your API key
}

# Make a GET request to the SerpAPI endpoint
response = requests.get("https://serpapi.com/search", params=params)

# Check for a successful response
if response.status_code == 200:
    data = response.json()
    print("Search Results:", data)
else:
    print(f"Error: {response.status_code}, {response.text}")

print("DONE")