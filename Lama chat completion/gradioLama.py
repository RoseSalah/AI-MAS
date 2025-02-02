import gradio as gr
from huggingface_hub import InferenceClient
import config

# Initialize the InferenceClient with your API key
client = InferenceClient(api_key=config.HF_KEY)

SYSTEM_PROMPT = """
Your role is market analyst, 
I want to provide you with a use case about specific business field ,
I want you to ask me the following questions one by one , keep in memory the answers I'll provide to you in order to use them to make the market study and the SWOT analysis based on internet research,
ask me about the location of the business , which sector , who are the potential customers and competitors , what is the nature / pattern of the business finally provide a market analysis study and SWOT analysis as well as competitors analysis based on all the information give a detailed market analysis , be specific to information provided and provide necessary information like the market size , Market trends and growth potential"""


# Function to interact with the model
def chat_with_model(user_input, history):
    # Convert chat history into the format expected by the model
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]  # Add system prompt
    for user_msg, model_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": model_msg})
    
    # Add the latest user input
    messages.append({"role": "user", "content": user_input})
    
    # Stream the response from the model
    stream = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct", 
        messages=messages, 
        max_tokens=200,
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
    chatbot = gr.Chatbot(label="Gemma-2-9b Chatbot")
    
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