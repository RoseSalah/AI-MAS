import gradio as gr
from huggingface_hub import InferenceClient
import requests
import config

# Initialize the InferenceClient with your API key
client = InferenceClient(api_key=config.HF_KEY)

# Bing Search API endpoint and key
BING_API_KEY = config.BING_API_KEY  # Replace with your Bing API key
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

SYSTEM_PROMPT = """
Your role is market analyst, 
I want to provide you with a use case about specific business field ,
I want you to ask me the following questions one by one , keep in memory the answers I'll provide to you in order to use them to make the market study and the SWOT analysis based on internet research,
ask me about the location of the business , which sector , who are the potential customers and competitors , what is the nature / pattern of the business finally provide a market analysis study and SWOT analysis as well as competitors analysis based on all the information give a detailed market analysis , be specific to information provided and provide necessary information like the market size , Market trends and growth potential

Please format your response as follows:
1. *Market Size*: Provide the market size, projections, and growth rate for the sector.
2. *Marketing Insights*:
   - Key Strategy: Summarize the main marketing strategy.
   - Suggested Platforms: List the preferred platforms for marketing.
   - Content Types: Mention the types of content that resonate with the target audience.
3. *SWOT Analysis*:
   Present the SWOT analysis in a *2x2 table* with the following columns:
   - *Strengths*
   - *Weaknesses*
   - *Opportunities*
   - *Threats*
4. *Competitor Overview*:
   Present the competitor analysis in a *table*, including:
   - *Competitor*: Name of the competitor.
   - *Market Share*: The competitor's market share percentage.
   - *Strengths*: Key strengths of the competitor.
   - *Weaknesses*: Weaknesses or challenges faced by the competitor.
5. *Customer Segments*:
   Provide a *pie chart* representation of the customer segments and their proportions. For each segment, include:
   - *Name*: The segment's name (e.g., 'Tech-Savvy Millennial').
   - *Demographics*: Age range, income level, location.
   - *Behavioral Traits*: Preferences, shopping habits.
   - *Pain Points*: Challenges or needs for this segment.
   - *Buying Motives*: Key factors driving purchasing decisions.
   """


# Function to perform Bing web search
def bing_search(query):
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": 3}  # Fetch top 5 results
    response = requests.get(BING_ENDPOINT, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to extract relevant information from Bing search results
def extract_search_results(search_results):
    if not search_results:
        return "No web results found."
    
    snippets = []
    for result in search_results.get("webPages", {}).get("value", []):
        snippets.append(f"- {result['snippet']}")
    
    return "\n".join(snippets)

# Function to interact with the model
def chat_with_model(user_input, history):
    # Check if the user wants to perform a web search
    if "use web search" in user_input.lower():
        # Extract the query from the user input
        query = user_input.replace("use web search", "").strip()
        
        # Perform Bing search
        search_results = bing_search(query)
        web_context = extract_search_results(search_results)
        
        # Add web context to the user input
        user_input = f"{user_input}\n\nWeb search results:\n{web_context}"
    
    # Convert chat history into the format expected by the model
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, model_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": model_msg})
    
    # Add the latest user input
    messages.append({"role": "user", "content": user_input})
    
    # Stream the response from the model
    stream = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct", 
        messages=messages, 
        max_tokens=500,
        stream=True
    )
    
    # Collect the response chunks
    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
    
    # Append the response to the chat history
    history.append((user_input, response))
    return history

# Custom Gradio Chat Interface using Blocks
with gr.Blocks() as demo:
    # Chat history component
    chatbot = gr.Chatbot(label="Llama 3.1 8B Chatbot")
    
    # Text input for user messages
    user_input = gr.Textbox(label="Your Message", placeholder="Type your message here...")
    
    # Submit button
    submit_button = gr.Button("Send")
    
    # Clear button to reset the chat
    clear_button = gr.Button("Clear Chat")
    
    # Function to handle user input and update chat history
    def respond(user_input, history):
        history = history or []  # Initialize history if None
        updated_history = chat_with_model(user_input, history)
        return updated_history, ""  # Return updated history and clear input box
    
    # Connect the submit button to the respond function
    submit_button.click(
        fn=respond,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    )
    
    # Clear the chat history
    def clear_chat():
        return None  # Reset chat history to empty
    
    clear_button.click(
        fn=clear_chat,
        inputs=[],
        outputs=chatbot
    )

# Launch the Gradio app
demo.launch()
