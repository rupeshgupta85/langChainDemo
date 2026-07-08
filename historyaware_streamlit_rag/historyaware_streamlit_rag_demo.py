import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Any
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="History-Aware RAG StreamlitChat",
    page_icon="💬",
    layout="wide"
)

# Load environment variables
load_dotenv()

GE_API_KEY = os.getenv("GE_API_KEY")

if not GE_API_KEY:
    st.error("GE_API_KEY environment variable is not set. Check your .env file.")
    st.stop()


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


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store_initialized" not in st.session_state:
    st.session_state.vector_store_initialized = False

if "uploaded_content" not in st.session_state:
    st.session_state.uploaded_content = None


def find_document_file():
    """Try multiple paths to find the document file."""
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of possible paths to try
    possible_paths = [
        # Same directory as script
        os.path.join(script_dir, "product-data.txt"),
        # In historyaware_rag subfolder (one level)
        os.path.join(script_dir, "historyaware_rag", "product-data.txt"),
        # In historyaware_streamlit_rag subfolder (one level)
        os.path.join(script_dir, "historyaware_streamlit_rag", "product-data.txt"),
        # Parent directory
        os.path.join(os.path.dirname(script_dir), "product-data.txt"),
        # Parent directory with subfolder
        os.path.join(os.path.dirname(script_dir), "historyaware_rag", "product-data.txt"),
        # Current working directory
        os.path.join(os.getcwd(), "product-data.txt"),
        # Relative paths
        "product-data.txt",
        "historyaware_rag/product-data.txt",
        "historyaware_streamlit_rag/product-data.txt",
    ]
    
    # Try each path
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


@st.cache_resource
def initialize_components(_content: str = None):
    """Initialize embeddings, LLM, and vector store (cached)."""
    
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
    
    # Initialize content variable
    content = ""
    documents = None
    
    # If content is provided (from upload), use it
    if _content:
        content = _content
        documents = [Document(page_content=content, metadata={"source": "uploaded-file"})]
    else:
        # Try to find the file
        file_path = find_document_file()
        
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                documents = [Document(page_content=content, metadata={"source": file_path})]
            except Exception as e:
                raise Exception(f"Error reading file at {file_path}: {str(e)}")
        else:
            raise FileNotFoundError(
                "Could not find product-data.txt. Please upload a document using the sidebar."
            )
    
    # Split document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create Chroma vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    
    retriever = vector_store.as_retriever()
    
    return embeddings, llm, retriever, len(chunks), len(content)


# Streamlit UI - Sidebar for file upload
st.sidebar.title("📁 Document Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload a text document (optional)",
    type=["txt"],
    help="If product-data.txt is not found, you can upload a document here"
)

if uploaded_file is not None:
    # Read the uploaded file
    content = uploaded_file.read().decode("utf-8")
    st.session_state.uploaded_content = content
    st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")

st.sidebar.divider()

# Initialize components with proper error handling
try:
    if st.session_state.uploaded_content:
        embeddings, llm, retriever, num_chunks, doc_size = initialize_components(
            st.session_state.uploaded_content
        )
    else:
        embeddings, llm, retriever, num_chunks, doc_size = initialize_components()
    
    st.session_state.vector_store_initialized = True
    
except FileNotFoundError as e:
    st.error(f"❌ {str(e)}")
    st.info("""
    **Please do one of the following:**
    1. Upload a text file using the sidebar, OR
    2. Place `product-data.txt` in one of these locations:
       - Same folder as the script
       - `historyaware_rag/` subfolder
       - `historyaware_streamlit_rag/` subfolder
    """)
    st.stop()
except Exception as e:
    st.error(f"❌ Error initializing components: {str(e)}")
    st.stop()


# Create contextualize question prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "Given a chat history and the latest user question "
         "which might reference context in the chat history, "
         "formulate a standalone question which can be understood "
         "without the chat history. Do NOT answer the question, "
         "just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Create contextualize chain
contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()


def contextualize_question(inputs: Dict[str, Any]) -> str:
    """Contextualize the question based on chat history."""
    chat_history = inputs.get("chat_history", [])
    question = inputs.get("input", "")
    
    if not chat_history:
        return question
    
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
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


def create_qa_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Create inputs for the QA chain."""
    contextualized_q = contextualize_question(inputs)
    docs = retrieve_docs(contextualized_q)
    context = format_docs(docs)
    
    return {
        "context": context,
        "chat_history": inputs.get("chat_history", []),
        "input": inputs.get("input", "")
    }


# Build the complete RAG chain
rag_chain = (
    RunnableLambda(create_qa_inputs)
    | qa_prompt
    | llm
    | StrOutputParser()
)


# Streamlit UI
st.title("💬 History-Aware RAG Streamlit Chat")
st.caption("Chat with your documents using context-aware retrieval")

# Sidebar
with st.sidebar:
    st.header("📊 System Info")
    st.metric("Document Size", f"{doc_size:,} chars")
    st.metric("Total Chunks", num_chunks)
    st.metric("Chat Messages", len(st.session_state.messages))
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.markdown("### 🔧 Model Info")
    st.text("LLM: Claude Sonnet 4.5")
    st.text("Embeddings: Titan v2")
    st.text("Provider: Generative Engine")
    
    st.divider()
    
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. Upload a document or ensure product-data.txt exists
        2. Type your question in the chat input
        3. The system maintains conversation context
        4. Ask follow-up questions naturally
        5. Click "Clear Chat History" to start over
        """)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Show searching indicator
            with st.spinner("🔍 Searching documents..."):
                # Invoke the RAG chain
                answer = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
            
            # Display answer
            message_placeholder.markdown(answer)
            
            # Update chat history
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=answer))
            
            # Add assistant message to display
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_message = f"❌ Error: {str(e)}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# Footer
st.divider()
st.caption("Powered by Generative Engine API | Built with Streamlit & LangChain")