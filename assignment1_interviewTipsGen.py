import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv

# Set your Bearer token directly (or use .env)
#API_TOKEN = ""

load_dotenv()  # Load environment variables from .env file

API_TOKEN = os.getenv("OPENAI_API_KEY")


prompt = PromptTemplate(
    input_variables=["company", "position", "strengths", "weaknesses"],
    template="""
You are a career coach. Provide tailored interview tips for the position of {position} at {company}.
Highlight your strengths in {strengths} and prepare for questions about your weaknesses such as {weaknesses}.
"""
)

llm = ChatOpenAI(
    model="amazon.nova-pro-v1:0",  # or whatever model your endpoint supports
    api_key=API_TOKEN,
    base_url="https://openai.generative.engine.capgemini.com/v1"  # ✅ important
)

st.title("Interview Tips Generator")

company = st.text_input("Company Name")
position = st.text_input("Position Title")
strengths = st.text_area("Your Strengths", height=100)
weaknesses = st.text_area("Your Weaknesses", height=100)

if company and position and strengths and weaknesses:
    response = llm.invoke(
        prompt.format(
            company=company,
            position=position,
            strengths=strengths,
            weaknesses=weaknesses
        )
    )

    st.write(response.content)