import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

# Get the API key from the .env file
api_key = os.getenv("PPLX_API_KEY")

# Create an instance of the OpenAI class
client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

# Construct the prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": "how's the weather in rockville, md at the moment?",
    },
]

response = client.chat.completions.create(
    model="pplx-70b-online",  # Replace with the appropriate model name
    messages=messages,
    max_tokens=240,
)

print(response.choices[0].message.content)
