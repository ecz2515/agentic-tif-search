import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def humanize_query_result(nl_query: str, sql_query: str, query_result) -> str:
    """
    Convert a SQL query result into a human-readable response based on the original natural language query.
    
    Args:
        nl_query: The original natural language query
        sql_query: The SQL query that was executed
        query_result: The result of the SQL query
        
    Returns:
        A human-readable response
    """
    try:
        # Format the query result for the LLM
        result_str = str(query_result)
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": """
                You are a helpful assistant that converts SQL query results into natural language responses.
                Your task is to take a SQL query, its result, and the original question, and provide a clear,
                concise, and natural language answer that directly addresses the user's question.
                
                Format currency values with dollar signs and commas (e.g., $1,234,567).
                Round large numbers to make them more readable.
                Use natural language that a non-technical person would understand.
                """},
                {"role": "user", "content": f"""
                Original question: {nl_query}
                
                SQL query executed: {sql_query}
                
                Query result: {result_str}
                
                Please provide a natural language answer to the original question based on this data.
                """}
            ],
            temperature=0.7,
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating humanized response: {e}" 