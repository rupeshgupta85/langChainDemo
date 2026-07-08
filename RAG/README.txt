
Make it Configurable (Best Practice)
Add to your .env file:

GE_API_KEY=your_api_key_here
DOCUMENT_PATH=C:/Users/rupeshgu/projects/langChainDemo/RAG/product-data.txt

This will work regardless of where you run the script from, as long as the folder structure is:

langChainDemo/
├── rag_demo.py (your script)
├── RAG/
│   └── product-data.txt
└── .env

=============

Output 
----
Loaded document from: c:\Users\rupeshgu\projects\langChainDemo\RAG\RAG\product-data.txt
Document size: 4300 characters
Document split into 5 chunks
Creating vector store (this may take a moment)...
Vector store created successfully!

=== Chat With Document ===
Type 'exit' or 'quit' to end the conversation

Your Question: How can I update my delivery address?

Searching documents...

Answer:
You can update your delivery address by going to **'Account Settings'** and selecting **'Edit Address'** under the **'Addresses'** section.

If your order is already in transit, please contact our support team for assistance.

Your Question: okay

Searching documents...

Answer:
Hello! I'm here to help you with any questions you might have about products, orders, shipping, returns, or your account. 

What can I assist you with today?

Your Question: 