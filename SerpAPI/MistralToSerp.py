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
    
    # Prepare the prompt with search context and the desired response format
    enriched_prompt = (
        f"Using the following context from multiple search engines:\n{search_snippets}\n"
        f"Answer this query:\n{user_query}\n"
        f"Please format your response as follows:\n"
        f"1. **Market Size**: Provide the market size, projections, and growth rate for the sector.\n"
        f"2. **Marketing Insights**: \n"
        f"   - Key Strategy: Summarize the main marketing strategy.\n"
        f"   - Suggested Platforms: List the preferred platforms for marketing.\n"
        f"   - Content Types: Mention the types of content that resonate with the target audience.\n"
        f"3. **SWOT Analysis**: \n"
        f"   Present the SWOT analysis in a **2x2 table** with the following columns:\n"
        f"   - **Strengths**\n"
        f"   - **Weaknesses**\n"
        f"   - **Opportunities**\n"
        f"   - **Threats**\n"
        f"4. **Competitor Overview**: \n"
        f"   Present the competitor analysis in a **table**, including:\n"
        f"   - **Competitor**: Name of the competitor.\n"
        f"   - **Market Share**: The competitor's market share percentage.\n"
        f"   - **Strengths**: Key strengths of the competitor.\n"
        f"   - **Weaknesses**: Weaknesses or challenges faced by the competitor.\n"
        f"5. **Customer Segments**: \n"
        f"   Provide a **pie chart** representation of the customer segments and their proportions. For each segment, include:\n"
        f"   - **Name**: The segment's name (e.g., 'Tech-Savvy Millennial').\n"
        f"   - **Demographics**: Age range, income level, location.\n"
        f"   - **Behavioral Traits**: Preferences, shopping habits.\n"
        f"   - **Pain Points**: Challenges or needs for this segment.\n"
        f"   - **Buying Motives**: Key factors driving purchasing decisions.\n"
        f"Answer is:"
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
