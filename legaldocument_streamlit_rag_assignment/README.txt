Key Features Added:

Streamlit UI Components:

Chat interface with message history
Sidebar with system metrics
Clear chat button
Loading spinners
Session State Management:

chat_history: Stores conversation context for the RAG chain
messages: Stores display messages for the UI
vector_store_initialized: Tracks initialization status
Caching with @st.cache_resource:

Embeddings, LLM, and vector store are initialized once
Improves performance on page reloads
Error Handling:

User-friendly error messages
Graceful degradation if files are missing

To Run:
================
# Install Streamlit
pip install streamlit

# Run the app
streamlit run your_script_name.py