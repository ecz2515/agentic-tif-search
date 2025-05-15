import os
import time
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def query_pdf_index(question: str, persist_dir="vectorstore", top_k=200):
    start_time = time.time()
    
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)
    query_engine = index.as_query_engine(
        llm=OpenAI(model="gpt-4o-mini", api_key=api_key),
        similarity_top_k=top_k,
        embed_model=embed_model
    )
    response = query_engine.query(question)
    
    end_time = time.time()
    print(f"PDF query took {end_time - start_time:.2f} seconds")
    
    return str(response)

# FOR DEBUGGING: Create a retriever to see what nodes are being fetched
# retriever = index.as_retriever(similarity_top_k=50)
# nodes = retriever.retrieve("How many new Red Line stations are they building?")

# Print out the nodes to see if they contain the information about four stations
# for i, node in enumerate(nodes):
#     print(f"Node {i}:\n{node.node.get_content()}\n")

# Hybrid retrieval - vector and keyword
# https://docs.llamaindex.ai/en/stable/examples/query_engine/CustomRetrievers/