import os
import requests
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()

SERP_API_KEY = os.getenv("SERP_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

mistral_client = Mistral(api_key=MISTRAL_API_KEY)
SERP_SEARCH_URL = "https://serpapi.com/search"

# Function to fetch search results from SerpAPI for a single engine
def fetch_serp_search_results(query, engine, count=4):
    params = {
        "engine": engine,   
        "q": query,         
        "num": count,       
        "api_key": SERP_API_KEY
    }
    response = requests.get(SERP_SEARCH_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
        if "organic_results" in results:
            return [
                {"name": item["title"], "url": item["link"]}
                for item in results["organic_results"]
            ]
        else:
            return []
    else:
        raise Exception(f"SerpAPI failed for engine '{engine}': {response.status_code} - {response.text}")


def fetch_results_from_multiple_engines(query, engines, count=3):
    combined_results = []
    for engine in engines:
        try:
            results = fetch_serp_search_results(query, engine, count)
            combined_results.extend(results)
        except Exception as e:
            print(f"Error fetching results for engine '{engine}': {e}")
    return combined_results

# Function to combine search results with user query and get a response from Mistral
def enhanced_query_with_search(user_query, engines=["google", "bing"]):
    # Fetch search results from multiple engines
    search_results = fetch_results_from_multiple_engines(user_query, engines)
    
    # Format the search results into a string
    search_snippets = "\n".join([f"{result['name']}: {result['url']}" for result in search_results])
    
    # Prepare the prompt with search context
    enriched_prompt = (
        f"Using the following context from multiple search engines:\n{search_snippets}\n"
        f"Answer this query:\n{user_query}\nAnswer is:"
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
        query = "How can you help me to conduct a market research?"
        engines = ["google", "bing"] 
        result = enhanced_query_with_search(query, engines)
        print(result)
    except Exception as e:
        print("Error:", e)
