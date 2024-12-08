import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
FALCON_API_KEY = os.getenv("FALCON_API_KEY")
# Hugging Face API link
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"  

def query_falcon_model(prompt, max_tokens=200, temperature=0.7):
    """
    Makes a call to the Hugging Face Falcon model API and returns the response.
    
    Args:
        prompt (str): The text prompt to send to the model.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): Sampling temperature.
    
    Returns:
        dict: The response from the model.
    """
    headers = {"Authorization": f"Bearer {FALCON_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature
        }
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to query model: {response.status_code} - {response.text}")

# Example usage
if __name__ == "__main__":
    prompt = "Explain the market size of the Agriculture market in Saudi Arabia."
    try:
        result = query_falcon_model(prompt)
        print("Response:", result)
    except Exception as e:
        print("Error:", e)
