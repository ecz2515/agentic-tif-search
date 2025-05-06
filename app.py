import os
from dotenv import load_dotenv
import openai

from db.init_duckdb import load_expenditures_table
from runner.executor import run_query
from llm.query_planner import generate_sql_from_nl
from llm.result_humanizer import humanize_query_result
from pdf_index.query_pdf import query_pdf_index
from pdf_index.vector_index import build_pdf_index

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

def classify_query(nl_query: str) -> str:
    """
    Use GPT to classify whether a query is best answered by SQL, PDF, or neither.

    Returns:
        "sql" | "pdf" | "none"
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are a classifier that decides whether a user's question should be answered using:\n"
                    "- SQL database queries (\"sql\")\n"
                    "- PDF documents and reports (\"pdf\")\n"
                    "- Or is conversational and doesn't require structured data (\"none\")\n\n"
                    "Return only one word: \"sql\", \"pdf\", or \"none\"\n\n"
                    "Examples:\n"
                    "- \"How much did Kinzie spend in 2023?\" → sql\n"
                    "- \"What are the goals of the Kinzie TIF?\" → pdf\n"
                    "- \"Thanks, that's helpful\" → none\n"
                )},
                {"role": "user", "content": nl_query}
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Classification error: {e}")
        return "sql"  # default to sql on error

def ensure_pdf_index_exists(persist_dir="vectorstore"):
    """
    Check if PDF vector index exists. If not, build it.
    """
    if not os.path.exists(os.path.join(persist_dir, "docstore.json")):
        print("No vector index found. Building one from PDFs...")
        build_pdf_index(pdf_dir="pdfs", persist_dir=persist_dir)
    else:
        print("PDF vector index found.")

def main():
    print("Loading DuckDB with TIF expenditure data...")
    con = load_expenditures_table()
    ensure_pdf_index_exists()

    print("Ask a question (e.g., 'How much did Kinzie spend in 2023?' or 'Who was the TIF administrator in 2022?')")
    while True:
        nl_query = input("\nQuestion (or type 'exit'): ").strip()
        if nl_query.lower() == "exit":
            break

        route = classify_query(nl_query)

        if route == "sql":
            print("[Primary Source: SQL]")
            try:
                sql = generate_sql_from_nl(nl_query)
                print(f"\nGenerated SQL:\n{sql}")
                result = run_query(con, sql)

                if not result or result.empty:
                    raise ValueError("No rows returned from SQL.")

                print("Result:\n", result)
                humanized_response = humanize_query_result(nl_query, sql, result)
                print("\nAnswer:", humanized_response)

            except Exception as e:
                print(f"SQL failed or empty: {e}")
                print("[Fallback Source: PDF]")
                answer = query_pdf_index(nl_query)
                print("\nAnswer:", answer)

        elif route == "pdf":
            print("[Source: PDF]")
            print("\nSearching reports for an answer...")
            answer = query_pdf_index(nl_query)
            print("\nAnswer:", answer)

        else:
            print("[Source: None]")
            print("\nAnswer: (no structured data used)")
            print(nl_query)


if __name__ == "__main__":
    main()
