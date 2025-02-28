import gradio as gr
from huggingface_hub import InferenceClient
import requests
import config

# Initialize the InferenceClient with your API key
client = InferenceClient(api_key=config.HF_KEY)

# Bing Search API endpoint and key
BING_API_KEY = config.BING_API_KEY  
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"


# Token limit constants
MAX_TOTAL_TOKENS = 4096  
SAFE_BUFFER = 500  #a buffer to prevent exceeding limits

# System prompt
SYSTEM_PROMPT = """
You are a market analyst. First, greet the user appropriately if they say "hi," "hello," "how are you?" or any similar greeting. Respond in a friendly and professional manner and ask if they need help in their business. If they ask how you are, reply briefly and then transition smoothly into the business discussion.  
### Instructions:
2. **Ask only one question at a time**, wait for the user's response, and then move to the next question.
3. **Do not generate detailed explanations** before moving to the next question. Keep responses concise unless specifically asked for details.

Business Analysis Questions you should ask in order:
1.Business Location.
2.Sector/Industry – Adapt phrasing, like "What field is your startup in?" or "Which industry are you focusing on?"
3.Potential Customers.
4.Business Model – Keep it natural, ask about how the user plans to operate and the nature of their business.
5.Unique advantages that the business have over competitors.

Market Analysis Output Format:
Based on the user's responses, format the output as follows:

1. *Market Size*: Provide the market size, projections, and growth rate for the sector.
2. *Marketing Insights*:
   - Key Strategy: Summarize the main marketing strategy.
   - Suggested Platforms: List the preferred platforms for marketing.
   - Content Types: Mention the types of content that resonate with the target audience.
3. *SWOT Analysis*:
   Present the SWOT analysis in a *2x2 table* with the following columns, provide detailed information :
   - *Strengths*
   - *Weaknesses*
   - *Opportunities*
   - *Threats*
4. *Competitor Overview*:
   **Real-World Competitor Analysis** *(Table)* show at least 4 real-world competitors, including:
   - *Competitor*: Name of the competitor.
   - *Market Share*: The competitor's market share percentage.
   - *Strengths*: Key strengths of the competitor.
   - *Weaknesses*: Weaknesses or challenges faced by the competitor.
5. *Customer Segments* including:
   - *Name*: The segment's name (e.g., 'Tech-Savvy Millennial').
   - *Demographics*: Age range, income level, location.
   - *Behavioral Traits*: Preferences, shopping habits.
   - *Pain Points*: Challenges or needs for this segment.
   - *Buying Motives*: Key factors driving purchasing decisions.Keep responses structured and precise.

Be professional and conversational, Avoid repetitive phrasing and rigid structures.
Use follow-ups and rephrase when necessary to keep the interaction natural. do not provide market details before finishing all questions.
If the user says "I don't know" or "I haven’t done enough research," take the initiative to conduct the necessary research.
After providing the analysis, ask the user if they found it helpful and how else you can assist

"""
# Function to perform Google Cloud Search

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
        return ""
    
    snippets = [f"- {result['snippet']}" for result in search_results.get("webPages", {}).get("value", [])]
    return "\n".join(snippets)


# Limit chat history to avoid exceeding token limits
def limit_history(history, max_entries=4):  
    return history[-max_entries:]

# Chat function with dynamic token adjustment
def chat_with_model(user_input, history):
    history = limit_history(history)  # Trim history to fit within token limits

    # Perform Google Cloud Search for every user input
    search_results = bing_search(user_input)
    web_context = extract_search_results(search_results)
    
    # Add web context to the user input
    user_input_with_context = f"{user_input}\n\nWeb search results:\n{web_context}"

    # Convert chat history into the format expected by the model
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, model_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": model_msg})

    # Append the user input with web context
    messages.append({"role": "user", "content": user_input_with_context})

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
        clear_button = gr.Button("🗑", elem_id="clear-button", scale=1)

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
