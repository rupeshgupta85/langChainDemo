import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

API_TOKEN = os.getenv("OPENAI_API_KEY")

def create_embedding(text, model, api_key):
    """Create an embedding using the OpenAI client."""
    try:
        client = OpenAI(
            base_url="https://openai.generative.engine.capgemini.com/v1",
            api_key=api_key
        )
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as err:
        print(f"Error occurred: {err}")
        return None

# Usage
text = input("Enter the text: ")
embedding = create_embedding(text, 'amazon.titan-embed-text-v2:0', API_TOKEN)

if embedding:
    print(f"Embedding generated successfully!")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
else:
    print("Failed to generate embedding")