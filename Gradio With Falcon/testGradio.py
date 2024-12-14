import gradio as gr
from falconToBingIntegration import enhanced_query_with_search

conversation_state = {
    "step": 1,              # Current step in the conversation
    "user_query": None,     # Stores the initial user query
    "country": None,        # Stores the country of interest
    "sector": None,         # Stores the sector of specialization
    "service_pattern": None, # Brief description of the service pattern
    "potential_customers": None, # User's input for potential customers
    "potential_competitors": None # User's input for potential competitors
}

def model_response(message, history):
    global conversation_state

    if conversation_state["step"] == 1:
        # Step 1: Initial query
        conversation_state["user_query"] = message
        conversation_state["step"] = 2
        response_text = enhanced_query_with_search(message)
        extracted_answer = response_text.split("Answer is: ")[1].strip()


        return extracted_answer + f" ,could you please specify the country where you want to establish your business?"

    elif conversation_state["step"] == 2:
        # Step 2: Collect country information
        conversation_state["country"] = message
        conversation_state["step"] = 3

        return (
            f"Thank you! Now, could you tell me which sector you are going to specialize in?"
        )

    elif conversation_state["step"] == 3:
        # Step 3: Collect sector information
        conversation_state["sector"] = message
        conversation_state["step"] = 4
        return (
            f"Great! Could you provide a brief description of your service pattern?"
        )

    elif conversation_state["step"] == 4:
        # Step 4: Collect service pattern
        conversation_state["service_pattern"] = message
        conversation_state["step"] = 5
        return (
            f"Got it! Who are your potential customers?"
        )

    elif conversation_state["step"] == 5:
        # Step 5: Collect potential customers
        conversation_state["potential_customers"] = message
        conversation_state["step"] = 6
        return (
            f"Understood! Now, who are your potential competitors?"
        )

    elif conversation_state["step"] == 6:
        # Step 6: Collect potential competitors
        conversation_state["potential_competitors"] = message

        # Combine all inputs to create a detailed query
        enriched_query = (
            f"Perform a market analysis for the following inputs:\n\n"
            f"Country: {conversation_state['country']}\n"
            f"Sector: {conversation_state['sector']}\n"
            f"Service Pattern: {conversation_state['service_pattern']}\n"
            f"Potential Customers: {conversation_state['potential_customers']}\n"
            f"Potential Competitors: {conversation_state['potential_competitors']}\n\n"
            f"Include the market size of the sector, opportunities, and a SWOT analysis."
        )

        # Query the model for market analysis
        analysis = enhanced_query_with_search(enriched_query)

        # Reset conversation state for new interactions
        conversation_state["step"] = 1
        conversation_state["user_query"] = None
        conversation_state["country"] = None
        conversation_state["sector"] = None
        conversation_state["service_pattern"] = None
        conversation_state["potential_customers"] = None
        conversation_state["potential_competitors"] = None

        return (
            f"Here is the market analysis based on your inputs:\n\n{analysis}"
        )

# Create Gradio interface
demo = gr.ChatInterface(model_response, type="messages").launch()

if __name__ == "__main__":
    demo.launch()


