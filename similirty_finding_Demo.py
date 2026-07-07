import os
import numpy as np
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
text1 = input("Enter the first text: ")
text2 = input("Enter the second text: ")
embedding1 = create_embedding(text1, 'amazon.titan-embed-text-v2:0', API_TOKEN)
embedding2 = create_embedding(text2, 'amazon.titan-embed-text-v2:0', API_TOKEN)

if embedding1 and embedding2:
    print(f"Embeddings generated successfully!")
    similirty_score = np.dot(embedding1, embedding2)
    print(f"Similarity score: {similirty_score*100, '%'}")
else:
    print("Failed to generate embedding")