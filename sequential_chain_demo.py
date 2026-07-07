import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Set your Bearer token directly (or use .env)
#API_TOKEN = ""

load_dotenv()  # Load environment variables from .env file

API_TOKEN = os.getenv("OPENAI_API_KEY")

# LLM
llm = ChatOpenAI(
    model="amazon.nova-pro-v1:0",  # or whatever model your endpoint supports
    api_key=API_TOKEN,
    base_url="https://openai.generative.engine.capgemini.com/v1"  # ✅ important
)

# Prompt 1: Generate Blog Outline
outline_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
You are a professional blogger.

Create an outline for a blog post on the following topic: {topic}

The outline should include:
- Introduction
- 3 main points with subpoints
- Conclusion
"""
)

# Prompt 2: Generate Introduction
introduction_prompt = PromptTemplate(
    input_variables=["outline"],
    template="""
You are a professional blogger.

Write an engaging introduction paragraph based on the following outline:

{outline}

The introduction should hook the reader and provide a brief overview of the topic.
"""
)

# Chain 1: Topic -> Outline
first_chain = outline_prompt | llm | StrOutputParser() | (lambda title: (st.write(title), title) [1])

# Chain 2: Outline -> Introduction
second_chain = introduction_prompt | llm

# Combined Chain
overall_chain = first_chain | second_chain

# Streamlit UI
st.title("Blog Post Generator")

topic = st.text_input("Input Topic")

if topic:
    response = overall_chain.invoke({"topic": topic})
    st.write(response.content)
    
    
# To run the Application 
#streamlit run .\sequential_chain_demo.py    