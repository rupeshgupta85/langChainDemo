
import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
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

#question = input("Enter the question you want to ask: ")
st.title("Ask anything.....")
question = st.text_input("Enter the question you want to ask:")


if question:
    response = response = llm.invoke(question)
    st.write(response.content)
