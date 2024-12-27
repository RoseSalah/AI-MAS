import os
import requests
from dotenv import load_dotenv
from mistralai import Mistral

# Load environment variables
load_dotenv()

# Get API keys from .env
BING_API_KEY = os.getenv("BING_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Initialize Mistral client
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# API URLs
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"

# Function to fetch search results from Bing
def fetch_bing_search_results(query, count=3):
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": count}
    response = requests.get(BING_SEARCH_URL, headers=headers, params=params)
    
    if response.status_code == 200:
        results = response.json()
        if "webPages" in results:
            search_results = [
                {"name": item["name"], "url": item["url"]}
                for item in results["webPages"]["value"]
            ]
            return search_results
        else:
            raise Exception("No web pages found in search results.")
    else:
        raise Exception(f"Bing API failed: {response.status_code} - {response.text}")

# Function to combine Bing search results with user query and get a response from Mistral
def enhanced_query_with_search(user_query):
    # Fetch search results from Bing
    search_results = fetch_bing_search_results(user_query)
    
    # Format the search results into a string
    search_snippets = "\n".join([f"{result['name']}: {result['url']}" for result in search_results])
    
    # Prepare the prompt with search context
    enriched_prompt = (
        f"Using the following context from the web:\n{search_snippets}\n"
        f"Answer this query:\n{user_query}\n Answer is:"
    )
    
    # Send the enriched prompt to Mistral
    chat_response = mistral_client.chat.complete(
        model="mistral-large-latest",  # Use the correct model name
        messages=[{"role": "user", "content": enriched_prompt}]
    )
    
    # Extract and return the model's response
    return chat_response.choices[0].message.content.strip()

# Example usage
if __name__ == "__main__":
    try:
        query = "How old is Joe Biden?"
        result = enhanced_query_with_search(query)
        print(result)
    except Exception as e:
        print("Error:", e)
