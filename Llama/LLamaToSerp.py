import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SERP_API_KEY = os.getenv("SERP_API_KEY")
LLAMA_API_KEY = os.getenv("MODEL_API_KEY")

# API URLs
SERP_SEARCH_URL = "https://serpapi.com/search"
LLAMA_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct"


def fetch_serp_search_results(query, engine, count=4):
    """
    Fetch search results from a specified search engine using SERP API.
    """
    params = {
        "engine": engine,
        "q": query,
        "num": count,
        "api_key": SERP_API_KEY
    }
    response = requests.get(SERP_SEARCH_URL, params=params)

    if response.status_code == 200:
        results = response.json()
        return [
            {"name": item["title"], "url": item["link"]}
            for item in results.get("organic_results", [])
        ]
    else:
        raise Exception(f"SERP API failed for engine '{engine}': {response.status_code} - {response.text}")


def fetch_results_from_multiple_engines(query, engines, count=3):
    """
    Fetch search results from multiple search engines.
    """
    combined_results = []
    for engine in engines:
        try:
            results = fetch_serp_search_results(query, engine, count)
            combined_results.extend(results)
        except Exception as e:
            print(f"Error fetching results for engine '{engine}': {e}")
    return combined_results


def query_llama_model(prompt, max_tokens=200, temperature=0.7):
    """
    Query the Llama model using Hugging Face API.
    """
    headers = {"Authorization": f"Bearer {LLAMA_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature
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
    Combine SERP API search results with the user query and send it to the Llama model.
    """
    # Fetch search results
    search_results = fetch_results_from_multiple_engines(user_query, engines=['Google', 'Bing'])

    # Prepare search snippets
    search_snippets = "\n".join([f"{result['name']}: {result['url']}" for result in search_results])

    # Enrich the prompt with context and instructions
    enriched_prompt = (
        f"Using the following context from multiple search engines:\n{search_snippets}\n"
        f"Answer this query:\n{user_query}\n"
        f"Please format your response as follows:\n"
        f"1. **Market Size**: Provide the market size, projections, and growth rate for the sector.\n"
        f"2. **Marketing Insights**:\n"
        f"   - Key Strategy: Summarize the main marketing strategy.\n"
        f"   - Suggested Platforms: List the preferred platforms for marketing.\n"
        f"   - Content Types: Mention the types of content that resonate with the target audience.\n"
        f"3. **SWOT Analysis**:\n"
        f"   Present the SWOT analysis in a **2x2 table** with the following columns:\n"
        f"   - **Strengths**\n"
        f"   - **Weaknesses**\n"
        f"   - **Opportunities**\n"
        f"   - **Threats**\n"
        f"4. **Competitor Overview**:\n"
        f"   Present the competitor analysis in a **table**, including:\n"
        f"   - **Competitor**: Name of the competitor.\n"
        f"   - **Market Share**: The competitor's market share percentage.\n"
        f"   - **Strengths**: Key strengths of the competitor.\n"
        f"   - **Weaknesses**: Weaknesses or challenges faced by the competitor.\n"
        f"5. **Customer Segments**:\n"
        f"   Provide a **pie chart** representation of the customer segments and their proportions. For each segment, include:\n"
        f"   - **Name**: The segment's name (e.g., 'Tech-Savvy Millennial').\n"
        f"   - **Demographics**: Age range, income level, location.\n"
        f"   - **Behavioral Traits**: Preferences, shopping habits.\n"
        f"   - **Pain Points**: Challenges or needs for this segment.\n"
        f"   - **Buying Motives**: Key factors driving purchasing decisions.\n"
        f"Answer is:"
    )

    # Query the Llama model
    llama_response = query_llama_model(enriched_prompt)

    # Trim the response to only include text after "Answer is:"
    if "Answer is:" in llama_response:
        return llama_response.split("Answer is:")[1].strip()
    else:
        return "No valid response found after 'Answer is'. Please verify the API response."


# Example Usage
if __name__ == "__main__":
    query = "Do the market research for a social media platform made for Turkey only targeting teenagers?"
    try:
        result = enhanced_query_with_search(query)
        print(result)
    except Exception as e:
        print("Error:", e)
