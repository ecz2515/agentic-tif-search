from db.init_duckdb import load_expenditures_table
from runner.executor import run_query
from llm.query_planner import generate_sql_from_nl
from llm.result_humanizer import humanize_query_result
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def is_query_requiring_sql(nl_query: str) -> bool:
    """
    Use LLM to determine if the input is a query that requires SQL processing.
    
    Args:
        nl_query: The user's input
        
    Returns:
        True if the input is a query requiring SQL, False otherwise
    """
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": """
                You are a classifier that determines if a user input requires SQL processing.
                
                Your task is to analyze the input and determine if it's:
                1. A question or request that requires querying a database with SQL
                2. A conversational response that doesn't need database access
                
                Examples of inputs requiring SQL:
                - "How much did Kinzie spend in 2023?"
                - "Show me all TIF districts"
                - "What was the total expenditure in 2022?"
                
                Examples of inputs NOT requiring SQL:
                - "Thanks"
                - "Goodbye"
                - "That's helpful"
                - "I understand"
                
                Respond with ONLY "true" or "false".
                """},
                {"role": "user", "content": nl_query}
            ],
            temperature=0,
        )
        
        result = response.choices[0].message.content.strip().lower()
        return result == "true"
    except Exception as e:
        print(f"Error in query classification: {e}")
        # Default to treating it as a query if there's an error
        return True

def main():
    print("Loading DuckDB with TIF expenditure data...")
    con = load_expenditures_table()

    print("Ask a question (e.g., 'How much did Kinzie spend in 2023?')")
    while True:
        nl_query = input("\nQuestion (or type 'exit'): ")
        if nl_query.lower() == "exit":
            break

        # Check if this is a query requiring SQL processing
        if not is_query_requiring_sql(nl_query):
            # Handle conversational responses without SQL
            print("\nAnswer:", nl_query)
            continue

        sql = generate_sql_from_nl(nl_query)
        print(f"\nGenerated SQL:\n{sql}")

        result = run_query(con, sql)
        print("Result:\n", result)
        
        # Generate a humanized response
        humanized_response = humanize_query_result(nl_query, sql, result)
        print("\nAnswer:", humanized_response)

if __name__ == "__main__":
    main()
