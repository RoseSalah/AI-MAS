import gradio as gr
from huggingface_hub import InferenceClient
import requests
import config

# Initialize the InferenceClient with your API key
client = InferenceClient(api_key=config.HF_KEY)

# Bing Search API endpoint and key
BING_API_KEY = config.BING_API_KEY  # Replace with your Bing API key
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

# Token limit constants
MAX_TOTAL_TOKENS = 4096  
SAFE_BUFFER = 500  # Keep a buffer to prevent exceeding limits

# Optimized system prompt (shorter but still effective)
SYSTEM_PROMPT = """
You are a market analyst. Ask the following questions one by one,meaning you ask a question
and take user's answer then ask the next question, keep in memory the user's answers
The questions are:
- Business location?
- Sector?
- Potential customers
- Who are the competitors
- Business nature/pattern

Based on user's responses, please format your response as follows:
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
   **Real-World Competitor Analysis** *(Table)*, including:
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
   - *Buying Motives*: Key factors driving purchasing decisions.Keep responses structured and precise.

    **Use real-world examples whenever possible**. If competitor data is unavailable, search online.

   """

# Function to perform Bing web search
def bing_search(query):
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": 3}  # Fetch top 3 results
    response = requests.get(BING_ENDPOINT, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Extract relevant search results
def extract_search_results(search_results):
    if not search_results:
        return "No web results found."
    
    snippets = [f"- {result['snippet']}" for result in search_results.get("webPages", {}).get("value", [])]
    return "\n".join(snippets)

# Limit chat history to avoid exceeding token limits
def limit_history(history, max_entries=4):  
    return history[-max_entries:]

# Chat function with dynamic token adjustment
def chat_with_model(user_input, history):
    history = limit_history(history)  # Trim history to fit within token limits

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, model_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": model_msg})

    messages.append({"role": "user", "content": user_input})

    # Estimate token usage
    total_tokens = sum(len(m["content"].split()) for m in messages)  
    max_tokens = max(500, min(1500, MAX_TOTAL_TOKENS - total_tokens - SAFE_BUFFER))

    if max_tokens < 100:  
        history = limit_history(history, max_entries=3)  # Further trim history
        return chat_with_model(user_input, history)  # Retry with smaller history

    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        messages=messages,
        max_tokens=max_tokens,  
        stream=True
    )

    response = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    history.append((user_input, response))
    return history

# Gradio UI (Updated Layout)
with gr.Blocks(css=".chat-container { display: flex; flex-direction: column; height: 100vh; }") as demo:
    gr.Markdown("# 📊 Market Analysis AI")
    
    chatbot = gr.Chatbot(label="Llama 3.2 Market Analyst", container=True)

    with gr.Row():
        user_input = gr.Textbox(
            show_label=False, placeholder="Type your message here...", lines=1, scale=9
        )
        submit_button = gr.Button("➤", elem_id="send-button", scale=1)
        clear_button = gr.Button("🗑️", elem_id="clear-button", scale=1)

    def respond(user_input, history):
        history = history or []
        return chat_with_model(user_input, history), ""

    # Send message when clicking the button
    submit_button.click(fn=respond, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

    # Send message when pressing Enter
    user_input.submit(fn=respond, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    
    def clear_chat():
        return None

    clear_button.click(fn=clear_chat, inputs=[], outputs=chatbot)

demo.launch()
