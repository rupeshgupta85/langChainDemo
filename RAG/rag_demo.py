import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from typing import List

# Load environment variables
load_dotenv()

GE_API_KEY = os.getenv("GE_API_KEY")

if not GE_API_KEY:
    raise ValueError("GE_API_KEY environment variable is not set. Check your .env file.")


# Custom Embeddings using OpenAI client
class GenerativeEngineEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "amazon.titan-embed-text-v2:0"):
        self.client = OpenAI(
            base_url="https://openai.generative.engine.capgemini.com/v1",
            api_key=api_key
        )
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            try:
                response = self.client.embeddings.create(input=text, model=self.model)
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Error embedding document: {e}")
                embeddings.append([0.0] * 1024)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            response = self.client.embeddings.create(input=text, model=self.model)
            return response.data[0].embedding
        except Exception as e:
            print(f"Error embedding query: {e}")
            return [0.0] * 1024


# Initialize embeddings
embeddings = GenerativeEngineEmbeddings(api_key=GE_API_KEY)

# LLM
llm = ChatOpenAI(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    api_key=GE_API_KEY,
    base_url="https://openai.generative.engine.capgemini.com/v1",
    default_headers={"x-api-key": GE_API_KEY},
    temperature=0.3,
    max_tokens=512,
    timeout=30,
)

# Build path to document - UPDATED SECTION
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "RAG", "product-data.txt")

# Load document
try:
    with open("RAG/product-data.txt", "r", encoding="utf-8") as f:
        content = f.read()
    documents = [Document(page_content=content, metadata={"source": "product-data.txt"})]
    print(f"Loaded document from: {file_path}")
    print(f"Document size: {len(content)} characters")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    print("Please ensure product-data.txt is in the RAG folder.")
    exit(1)

# Split document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)

print(f"Document split into {len(chunks)} chunks")

# Create Chroma vector store
print("Creating vector store (this may take a moment)...")
try:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    print("Vector store created successfully!")
except Exception as e:
    print(f"Error creating vector store: {e}")
    exit(1)

retriever = vector_store.as_retriever()

# Format documents function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant.

Answer the user's question using only the supplied context.

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)

# Create RAG chain using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chat loop
print("\n=== Chat With Document ===")
print("Type 'exit' or 'quit' to end the conversation\n")

while True:
    question = input("Your Question: ")
    
    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    if not question.strip():
        continue
    
    try:
        print("\nSearching documents...")
        response = rag_chain.invoke(question)
        print("\nAnswer:")
        print(response)
        print()
    except Exception as e:
        print(f"\nError: {str(e)}\n")