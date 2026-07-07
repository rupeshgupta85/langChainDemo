
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Set your Bearer token directly (or use .env)
#API_TOKEN = ""

load_dotenv()  # Load environment variables from .env file

API_TOKEN = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="amazon.nova-pro-v1:0",  # or whatever model your endpoint supports
    api_key=API_TOKEN,
    base_url="https://openai.generative.engine.capgemini.com/v1"  # ✅ important
)

question = input("Enter the question you want to ask: ")
response = llm.invoke(question)

print(response.content)
