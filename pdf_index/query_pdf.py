import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def query_pdf_index(question: str, persist_dir="vectorstore", top_k=10):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(
        llm=OpenAI(model="gpt-4o", api_key=api_key),
        similarity_top_k=top_k
    )
    response = query_engine.query(question)
    return str(response)
