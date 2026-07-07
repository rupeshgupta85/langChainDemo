import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Generative Engine API Key
GE_API_KEY = os.getenv("GE_API_KEY")

if not GE_API_KEY:
    raise ValueError("GE_API_KEY environment variable is not set")

# Embeddings - Added x-api-key header
embeddings = OpenAIEmbeddings(
    model="amazon.titan-embed-text-v2:0",
    api_key=GE_API_KEY,
    base_url="https://openai.generative.engine.capgemini.com/v1",
    default_headers={"x-api-key": GE_API_KEY}
)

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

# Load document
try:
    loader = TextLoader("product-data.txt", encoding="utf-8")
    documents = loader.load()
except FileNotFoundError:
    print("Error: product-data.txt not found. Please create this file first.")
    exit(1)

# Split document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)

print(f"Document split into {len(chunks)} chunks")

# Create Chroma vector store
print("Creating vector store...")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)
retriever = vector_store.as_retriever()

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
        ("human", "{input}"),
    ]
)

# Build chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

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
        response = rag_chain.invoke({"input": question})
        print("\nAnswer:")
        print(response["answer"])
        print()
    except Exception as e:
        print(f"\nError: {str(e)}\n")