import random
import gradio as gr
from LLamaToSerp import enhanced_query_with_search  



def model_response(message, history):
    response_text= enhanced_query_with_search(message)
    return response_text


demo = gr.ChatInterface(model_response, type="messages").launch()

if __name__ == "__main__":
    demo.launch()



