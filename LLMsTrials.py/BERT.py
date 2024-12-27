import requests
import os
from dotenv import load_dotenv

load_dotenv()
BING_API_KEY = os.getenv("BING_API_KEY")
LLAMA_API_KEY = os.getenv("MODEL_API_KEY")

# API connections
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
LLAMA_API_URL = "https://api-inference.huggingface.co/models/bert-base-uncased"


def fetch_bing_search_results(query, count=3):
    """
    Fetches search results from Bing Search API.
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


def query_Llama_model(prompt, max_tokens=200, temperature=0.7):
    """
    Calls Hugging Face Llama 3.2 model API.
    """
    headers = {"Authorization": f"Bearer {LLAMA_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            # "max_new_tokens": max_tokens,
            # "temperature": temperature
        }
    }
    response = requests.post(LLAMA_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        response_json = response.json()
        
        if isinstance(response_json, list) and "generated_text" in response_json[0]:
            return response_json[0]["generated_text"]
        else:
            raise Exception(f"Unexpected response structure: {response_json}")
    else:
        raise Exception(f"Llama API failed: {response.status_code} - {response.text}")


def enhanced_query_with_search(user_query):
    """
    Combines Bing Search results with user query and sends to Llama model.
    """
    # Get search results
    search_results = fetch_bing_search_results(user_query)
    
    # Prepare search snippets
    search_snippets = "\n".join([f"{result['name']}: {result['url']}" for result in search_results])
    
    # Prepare enriched prompt
    enriched_prompt = (
        f"Using the following context from the web:\n{search_snippets}\n"
        f"Answer this query:\n{user_query}\n Answer is:"
    )
    
    # Query Falcon model
    falcon_response = query_Llama_model(enriched_prompt)
    
    # Format the model's response
    model_response = falcon_response.strip()
    formatted_response = (
        f"{model_response}, ....\n\n"  # Add the extra comma and text as per your requirement
        f"Searched sites:\n" +
        "\n".join([f"{i+1}. {result['name']}: {result['url']}" for i, result in enumerate(search_results)])
    )
    
    # Remove everything before "Answer is:"
    if "Answer is:" in formatted_response:
        formatted_response = formatted_response.split("Answer is:")[1].strip()
    
    return formatted_response


# Example Usage
if __name__ == "__main__":
    query = "What is the pregnancy length for rabbits?"
    try:
        result = enhanced_query_with_search(query)
        print(result)
    except Exception as e:
        print("Error:", e)
