import os
from dotenv import load_dotenv
import openai
import time
import sys
from datetime import datetime

from db.init_duckdb import load_expenditures_table
from llm.agent import TIFAgent
from pdf_index.vector_index import build_pdf_index

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80 + "\n")

def print_step(text: str):
    """Print a step in the process."""
    print(f"\n→ {text}")

def print_info(text: str):
    """Print informational text."""
    print(f"ℹ️  {text}")

def print_success(text: str):
    """Print success message."""
    print(f"✅ {text}")

def print_warning(text: str):
    """Print warning message."""
    print(f"⚠️  {text}")

def print_error(text: str):
    """Print error message."""
    print(f"❌ {text}")

def spinning_cursor():
    """Create a spinning cursor animation."""
    while True:
        for cursor in '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏':
            yield cursor

def print_with_spinner(text: str, spinner):
    """Print text with a spinning cursor."""
    sys.stdout.write(f"\r{next(spinner)} {text}")
    sys.stdout.flush()

def ensure_pdf_index_exists(persist_dir="vectorstore"):
    """
    Check if PDF vector index exists. If not, build it.
    """
    if not os.path.exists(os.path.join(persist_dir, "docstore.json")):
        print_warning("No vector index found. Building one from PDFs...")
        build_pdf_index(pdf_dir="pdfs", persist_dir=persist_dir)
        print_success("PDF vector index built successfully!")
    else:
        print_success("PDF vector index found and loaded.")

def main():
    print_header("TIF Data Analysis Assistant")
    print_info("Initializing system...")
    
    # Initialize database
    print_step("Loading DuckDB with TIF expenditure data...")
    ensure_pdf_index_exists()
    
    # Initialize the TIF agent
    print_step("Initializing AI agent...")
    agent = TIFAgent(client)
    print_success("System ready!")

    print_header("Interactive Session Started")
    print_info("You can ask questions about TIF expenditures and reports.")
    print_info("Example questions:")
    print("  • How much did Kinzie spend in 2023 and what were their main goals?")
    print("  • Compare LaSalle's spending in 2022 with their stated objectives")
    print("  • What were the major projects in Kinzie's 2023 report?")
    print("\nType 'exit' to end the session.")

    while True:
        print("\n" + "-" * 80)
        nl_query = input("\n📝 Your question: ").strip()
        
        if nl_query.lower() == "exit":
            print_header("Session Ended")
            print_info(f"Thank you for using the TIF Data Analysis Assistant!")
            break

        try:
            print_step("Processing your question...")
            spinner = spinning_cursor()
            
            # Start time for response
            start_time = time.time()
            
            # Process the query using our agent
            response = agent.process_query(nl_query)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Clear the spinner
            sys.stdout.write("\r" + " " * 100 + "\r")
            
            print_header("Response")
            print(response)
            print_info(f"Response generated in {response_time:.2f} seconds")
            
        except Exception as e:
            print_error(f"Error processing query: {e}")
            print_warning("Please try rephrasing your question or ask something else.")

if __name__ == "__main__":
    main()
