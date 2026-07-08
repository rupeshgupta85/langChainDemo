import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Any

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

# Build path to document
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "historyaware_streamlit_rag", "product-data.txt")

# Load document
try:
    with open("historyaware_streamlit_rag/product-data.txt", "r", encoding="utf-8") as f:
        content = f.read()
    documents = [Document(page_content=content, metadata={"source": "product-data.txt"})]
    print(f"Loaded document from: {file_path}")
    print(f"Document size: {len(content)} characters")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    print("Please ensure product-data.txt is in the historyaware_streamlit_rag folder.")
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

# Create contextualize question prompt for history-aware retrieval
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "Given a chat history and the latest user question "
         "which might reference context in the chat history, "
         "formulate a standalone question which can be understood "
         "without the chat history. Do NOT answer the question, "
         "just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create contextualize chain
contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()


def contextualize_question(inputs: Dict[str, Any]) -> str:
    """Contextualize the question based on chat history."""
    chat_history = inputs.get("chat_history", [])
    question = inputs.get("input", "")
    
    # If no chat history, return the question as is
    if not chat_history:
        return question
    
    # Otherwise, reformulate the question with context
    contextualized = contextualize_chain.invoke({
        "chat_history": chat_history,
        "input": question
    })
    return contextualized


def retrieve_docs(contextualized_question: str) -> List[Document]:
    """Retrieve relevant documents."""
    return retriever.invoke(contextualized_question)


def format_docs(docs: List[Document]) -> str:
    """Format documents for context."""
    return "\n\n".join(doc.page_content for doc in docs)


# Create the QA prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are a helpful assistant. "
         "Answer the user's question using only the supplied context. "
         "If you don't know the answer, say you don't know.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def create_qa_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Create inputs for the QA chain."""
    # Contextualize the question
    contextualized_q = contextualize_question(inputs)
    
    # Retrieve documents
    docs = retrieve_docs(contextualized_q)
    
    # Format context
    context = format_docs(docs)
    
    return {
        "context": context,
        "chat_history": inputs.get("chat_history", []),
        "input": inputs.get("input", "")
    }


# Build the complete RAG chain using LCEL
rag_chain = (
    RunnableLambda(create_qa_inputs)
    | qa_prompt
    | llm
    | StrOutputParser()
)

# Initialize chat history
chat_history = []

# Chat loop
print("\n=== Chat With Document (History-Aware) ===")
print("Type 'exit' or 'quit' to end the conversation")
print("Type 'clear' to clear chat history\n")

while True:
    question = input("Your Question: ")
    
    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    if question.lower() == "clear":
        chat_history = []
        print("Chat history cleared!\n")
        continue
    
    if not question.strip():
        continue
    
    try:
        print("\nSearching documents...")
        
        # Invoke the chain with chat history
        answer = rag_chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
        
        print("\nAnswer:")
        print(answer)
        print()
        
        # Update chat history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        
    except Exception as e:
        print(f"\nError: {str(e)}\n")