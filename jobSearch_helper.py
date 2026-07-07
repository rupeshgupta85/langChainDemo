import os

from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


class CapgeminiEmbeddings:
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts):
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), 10):
            batch = texts[i:i + 10]
            response = self.client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# Load .env
load_dotenv()

API_TOKEN = os.getenv("OPENAI_API_KEY")

if not API_TOKEN:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Create Embeddings Model
embeddings = CapgeminiEmbeddings(
    model="amazon.titan-embed-text-v2:0",
    api_key=API_TOKEN,
    base_url="https://openai.generative.engine.capgemini.com/v1"
)

# Load documents
documents = TextLoader("job_listings.txt").load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10
)

chunks = text_splitter.split_documents(documents)

# Create FAISS vector store
db = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Query
query = input("Enter the query: ")

# Search
results = db.similarity_search(query, k=3)

print("\nTop Matches:\n")

for i, doc in enumerate(results, start=1):
    print(f"Result {i}")
    print(doc.page_content)
    print("-" * 50)