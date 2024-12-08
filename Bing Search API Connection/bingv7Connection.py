import json
import os
from pprint import pprint
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
subscription_key = os.getenv("API_KEY")

endpoint = "https://api.bing.microsoft.com/v7.0/search" #+ "/bing/v7.0/search" => causes 404 resource not found error

# Query term to search for. 
query = "Microsoft Cognitive Services"

# Construct a request
mkt = 'en-US'
params = {'q': query, 'mkt': mkt}
headers = {'Ocp-Apim-Subscription-Key': subscription_key}

# Call the API
try:
    response = requests.get(endpoint, headers=headers, params=params)
    print(f"Requesting URL: {response.url}")
    response.raise_for_status()
    print("\nHeaders:\n")
    print(response.headers)
    print("\nJSON Response:\n")
    pprint(response.json())
    # print("ALL DONE GIRL")
except Exception as ex:
    print("An error occurred:", ex)