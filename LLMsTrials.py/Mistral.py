import os
from mistralai import Mistral

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": "can you help to conduct market research??",
        },
    ]
)
print(chat_response.choices[0].message.content)