Key Changes Made:

Removed dependency on langchain.chains - Built custom implementation using only langchain_core components

Created manual history-aware retrieval:

contextualize_question(): Reformulates questions based on chat history
retrieve_docs(): Retrieves relevant documents
format_docs(): Formats documents for context
create_qa_inputs(): Orchestrates the entire flow
Used LCEL (LangChain Expression Language):

RunnableLambda to wrap custom functions
Chain components using | operator
Maintains compatibility with LangChain's streaming and async features
Simplified dependencies: Only requires:

pip install langchain-core langchain-openai langchain-chroma langchain-text-splitters python-dotenv openai chromadb

How it works:

When a question comes in, it first checks if there's chat history
If yes, it uses the LLM to reformulate the question standalone
The reformulated question is used to retrieve relevant documents
Both the context and chat history are passed to the final QA chain
The answer is generated and added to chat history