import os
from mistralai import Mistral

# Use the API key directly
client = Mistral(api_key="your_api_key_here")

model = "mistral-large-latest"

chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ]
)
print(chat_response.choices[0].message.content)