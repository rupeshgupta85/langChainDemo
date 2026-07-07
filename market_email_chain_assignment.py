import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

load_dotenv()  # Load environment variables from .env file

API_TOKEN = os.getenv("OPENAI_API_KEY")

# LLM
llm = ChatOpenAI(
    model="amazon.nova-pro-v1:0",  # or whatever model your endpoint supports
    api_key=API_TOKEN,
    base_url="https://openai.generative.engine.capgemini.com/v1"  # ✅ important
)

# ------------------------------------------------------------------
# Prompt 1 - Subject Line Generator
# ------------------------------------------------------------------

subject_prompt = PromptTemplate(
    input_variables=["product_name", "features"],
    template="""
You are an experienced marketing specialist.

Create a catchy email subject line for the following product:

Product: {product_name}

Features:
{features}

Return only the subject line.
"""
)

# ------------------------------------------------------------------
# Prompt 2 - Email Generator
# ------------------------------------------------------------------

email_prompt = PromptTemplate(
    input_variables=[
        "subject_line",
        "product_name",
        "target_audience",
        "features"
    ],
    template="""
You are an expert email marketer.

Write a professional marketing email.

Subject:
{subject_line}

Product:
{product_name}

Features:
{features}

Target Audience:
{target_audience}

Requirements:
- 250 to 300 words
- Engaging opening
- Highlight key benefits
- Include call-to-action
- Professional tone

Return only the email body.
"""
)

# ------------------------------------------------------------------
# Chains
# ------------------------------------------------------------------

subject_chain = (
    subject_prompt
    | llm
    | StrOutputParser()
)

email_chain = (
    email_prompt
    | llm
    | StrOutputParser()
)

# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------

st.set_page_config(page_title="Marketing Email Generator")

st.title("📧 Marketing Email Generator")

product_name = st.text_input("Product Name")
features = st.text_area("Product Features")
target_audience = st.text_input("Target Audience")

if st.button("Generate Email"):

    if product_name and features and target_audience:

        subject_line = subject_chain.invoke(
            {
                "product_name": product_name,
                "features": features
            }
        )

        email_body = email_chain.invoke(
            {
                "subject_line": subject_line,
                "product_name": product_name,
                "features": features,
                "target_audience": target_audience
            }
        )

        st.subheader("Generated Subject Line")
        st.success(subject_line)

        st.subheader("Generated Email")
        st.write(email_body)

    else:
        st.warning("Please fill all fields.")