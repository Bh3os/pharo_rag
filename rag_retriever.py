import os
import argparse
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List
import time 


CHROMA_DB_PATH ="/content/chroma/chroma_db"
COLLECTION_NAME = "ancient_egypt_collection"
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
OPENROUTER_API_KEY = "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" # User provided API Key
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "meta-llama/llama-4-maverick"

def format_docs(docs: List[Document]) -> str:
    """Formats the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def print_retrieved_docs(docs: List[Document]) -> List[Document]:
    """Prints the retrieved documents and returns them."""
    print("\n--- Retrieved Documents ---")
    if not docs:
        print("No documents retrieved.")
    else:
        for i, doc in enumerate(docs):
            print(f"Document {i+1} (ID: {doc.metadata.get('id', 'N/A')}):")
            print(f"  Content: {doc.page_content[:300]}...") # Print snippet
    print("-------------------------")
    return docs

# Initialize components outside the function to make them globally accessible
print("--- Initializing RAG System Components Globally ---")

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"trust_remote_code": True}
)
print("Embedding model loaded.")

print(f"Loading Chroma vector store from: {CHROMA_DB_PATH}")
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)
print(f"Chroma vector store loaded with {vectorstore._collection.count()} documents.")

print(f"Initializing OpenRouter LLM: {OPENROUTER_MODEL}")
llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_BASE_URL,
    temperature=0.7
)
print("OpenRouter LLM initialized.")

template = """
    You are an assistant knowledgeable about ancient Egyptian history based on the provided context.
    Based on the following context about ancient Egyptian history, answer the question.
    Synthesize information from the context if necessary to provide a comprehensive answer.
    If you see mentions of dynasties, periods, or other historical elements in the context, organize and present them in a structured way.
    Even if the context only contains partial information, try to provide the most complete answer possible.
    If the context does not contain any relevant information to answer the question, state that.

    Context:
    {context}

    Question: {question}

    Answer:
    """

prompt = ChatPromptTemplate.from_template(template)
print("Prompt template created.")

print("--- Global Initialization Complete ---")


def ask_ancient_egypt_loop(top_k: int = 5):
    # Use the globally initialized vectorstore and components
    retriever_top_k = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # Build RAG Chain using globally initialized components
    print("Building RAG chain within the loop function...")
    rag_chain = (
        {"retrieved_docs": retriever_top_k, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(context=RunnableLambda(lambda x: print_retrieved_docs(x["retrieved_docs"])) | RunnableLambda(format_docs))
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain built.")


    print("Ask your questions about Ancient Egypt! (Type 'exit' or 'quit' to stop)\n")
    while True:
        query = input("🟡 Your question: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("🔚 Exiting RAG Q&A.")
            break

        print(f"\n--- Executing Query: {query} ---")
        start_time = time.process_time()
        try:
            response = rag_chain.invoke(query)
            end_time = time.process_time()
            print(f"\n🟢 Final Answer:\n{response}")
            print(f"\n⏱️ (Query took {end_time - start_time:.2f} seconds)\n")
        except Exception as e:
            print(f"❌ Error during query: {e}\n")

# Call the function to start the loop
ask_ancient_egypt_loop()