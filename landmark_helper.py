import base64
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from PIL import Image
from io import BytesIO

load_dotenv()  # Load environment variables from .env file

API_TOKEN = os.getenv("OPENAI_API_KEY")

def encode_image(uploaded_file):
    """Convert image to base64 and determine MIME type"""
    # Read the file
    file_bytes = uploaded_file.read()
    
    # Detect image format using PIL
    image = Image.open(BytesIO(file_bytes))
    image_format = image.format.lower()
    
    # Convert to JPEG if it's PNG or other format
    if image_format != 'jpeg' and image_format != 'jpg':
        # Convert to RGB (in case of RGBA or other modes)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as JPEG
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        file_bytes = buffer.getvalue()
        image_format = 'jpeg'
    
    # Encode to base64
    b64_string = base64.b64encode(file_bytes).decode("utf-8")
    
    return b64_string, image_format


# Generative Engine Vision Model using OpenAI-compatible endpoint
llm = ChatOpenAI(
    model="amazon.nova-pro-v1:0",  # Vision-capable model
    base_url="https://openai.generative.engine.capgemini.com/v1",
    api_key=API_TOKEN,
    temperature=0,
    max_tokens=1024
)

# Streamlit UI
st.title("Landmark Helper (Generative Engine)")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, width=400)

    # Encode image and get format
    image_b64, image_format = encode_image(uploaded_file)
    
    # Create dynamic prompt with correct MIME type
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that identifies landmarks from images."
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": """
                        Identify the landmark shown in the image.
                        Return:
                        1. Landmark name
                        2. Location
                        3. Short description
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{{image}}"
                        }
                    }
                ]
            )
        ]
    )
    
    chain = prompt | llm

    with st.spinner("Analyzing image..."):
        try:
            response = chain.invoke(
                {
                    "image": image_b64
                }
            )
            
            st.subheader("Result")
            st.write(response.content)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("The model may have specific image format requirements. The image has been automatically converted to JPEG format.")