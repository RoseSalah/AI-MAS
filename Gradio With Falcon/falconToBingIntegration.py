import requests
import os
from dotenv import load_dotenv

#load env variables
load_dotenv()
BING_API_KEY = os.getenv("BING_API_KEY")
FALCON_API_KEY = os.getenv("MODEL_API_KEY")
# API connections
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
FALCON_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"


def fetch_bing_search_results(query, count=3):
    """
    Fetches search results from Bing Search API.
    
    Args:
        query (str): The search query.
        count (int): Number of results to fetch.
    
    Returns:
        list: A list of dictionaries containing the name and URL of the search results.
    """
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": count}
    response = requests.get(BING_SEARCH_URL, headers=headers, params=params)
    if response.status_code == 200:
        results = response.json()        
        if isinstance(results, dict) and "webPages" in results and "value" in results["webPages"]:
            search_results = [
                {"name": item["name"], "url": item["url"]} 
                for item in results["webPages"]["value"]
            ]
            return search_results
        else:
            raise Exception("Unexpected response structure or no results found.")
    else:
        raise Exception(f"Bing API failed: {response.status_code} - {response.text}")


def query_falcon_model(prompt, max_tokens=200, temperature=0.7):
    """
    Calls Hugging Face Falcon model API.
    
    Args:
        prompt (str): The text prompt for the model.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
    
    Returns:
        str: Generated response from the model.
    """
    headers = {"Authorization": f"Bearer {FALCON_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature
        }
    }
    response = requests.post(FALCON_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        response_json = response.json()
        
        if isinstance(response_json, list) and "generated_text" in response_json[0]:
            return response_json[0]["generated_text"]
        else:
            raise Exception(f"Unexpected response structure: {response_json}")
    else:
        raise Exception(f"Falcon API failed: {response.status_code} - {response.text}")


def enhanced_query_with_search(user_query):
    """
    Combines Bing Search results with user query and sends to Falcon model.
    
    Args:
        user_query (str): The user's query.
    
    Returns:
        str: Final formatted answer to the user's question.
    """

    search_results = fetch_bing_search_results(user_query) # Get search results
    search_snippets = "\n".join([f"{result['name']}: {result['url']}" for result in search_results]) # Combine search results with user query
    enriched_prompt = f"Using the following context from the web:\n{search_snippets}\nAnswer this query:\n{user_query}\nAnswer is: "
    falcon_response = query_falcon_model(enriched_prompt) # Query Falcon model    
    formatted_response = (
        # f"Answering your question: \"{user_query}\"\n"
        # f"Searching web results:\n"
        # f"{search_snippets}\n\n"
        f"Model Response:\n{falcon_response.strip()}"
    )
    return formatted_response


# Example Usage
if __name__ == "__main__":
    query = "can you help me to conduct a market study analysis?"
    try:
        result = enhanced_query_with_search(query)
        print(result)
    except Exception as e:
        print("Error:", e)