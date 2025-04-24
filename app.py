from db.init_duckdb import load_expenditures_table
from runner.executor import run_query
from llm.query_planner import generate_sql_from_nl
from llm.result_humanizer import humanize_query_result

def main():
    print("Loading DuckDB with TIF expenditure data...")
    con = load_expenditures_table()

    print("Ask a question (e.g., 'How much did Kinzie spend in 2023?')")
    while True:
        nl_query = input("\nQuestion (or type 'exit'): ")
        if nl_query.lower() == "exit":
            break

        sql = generate_sql_from_nl(nl_query)
        print(f"\nGenerated SQL:\n{sql}")

        result = run_query(con, sql)
        print("Result:\n", result)
        
        # Generate a humanized response
        humanized_response = humanize_query_result(nl_query, sql, result)
        print("\nAnswer:", humanized_response)

if __name__ == "__main__":
    main()
