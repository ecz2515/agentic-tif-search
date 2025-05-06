import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def build_pdf_index(pdf_dir="pdfs", persist_dir="vectorstore"):
    documents = SimpleDirectoryReader(pdf_dir).load_data()
    embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"Vector index built and saved to {persist_dir}")
