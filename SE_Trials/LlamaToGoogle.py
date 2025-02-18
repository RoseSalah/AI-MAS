import gradio as gr
from huggingface_hub import InferenceClient
import requests
import config

# Initialize the InferenceClient with your API key
client = InferenceClient(api_key=config.HF_KEY)

# Google Cloud Search API endpoint and key
GCS_API_KEY = config.GCS_API_KEY  
GCS_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"
CX = config.GCS_CX  

# Token limit constants
MAX_TOTAL_TOKENS = 4096  
SAFE_BUFFER = 500  #a buffer to prevent exceeding limits

# System prompt
SYSTEM_PROMPT = """
You are an expert market analyst assisting users with business research.  
Your goal is to collect necessary details through natural, varied conversation, avoiding rigid question formats.  
Gather information dynamically based on context and user responses, adjusting your phrasing and approach.  

### **Key Information to Collect:**
1. **Business Location** – Ask casually, e.g., "Where do you plan to launch?" or "Which region are you targeting?"
2. **Sector/Industry** – Adapt phrasing, like "What field is your startup in?" or "Which industry are you focusing on?"
3. **Potential Customers & Competitors** – Avoid repetition, e.g., "Who would buy from you?" or "Who else offers similar products?"
4. **Business Model** – Keep it natural, e.g., "How do you plan to operate?" or "What's your revenue approach?"

### **Response Structure:**
After gathering enough details, provide:
- **Market Size** (growth trends, projections)
- **Marketing Insights** (effective strategies, ideal platforms)
- **SWOT Analysis** (Strengths, Weaknesses, Opportunities, Threats)
- **Competitor Overview** (comparison table with Name, Market Share, Strengths, Weaknesses)
- **Customer Segments** (visual breakdown of demographics, behaviors, pain points, buying motives)
If the answers gathered do not have enough information about potential customers or competitors, do your research and provide the data.

**Be conversational, adaptive, and professional.** Never repeat the same phrasing or structure for questions.  
Use follow-ups and rephrase when necessary to make interactions feel natural.
"""

# Function to perform Google Cloud Search

def gcs_search(query):
    params = {"q": query, "key": GCS_API_KEY, "cx": CX}
    response = requests.get(GCS_ENDPOINT, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Extract relevant search results
def extract_search_results(search_results):
    if not search_results or "items" not in search_results:
        return "No web results found."
    
    snippets = [f"- {item['snippet']}" for item in search_results["items"]]
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