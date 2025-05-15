import openai
from dotenv import load_dotenv
import os
from db.schema_discovery import get_prompt_context

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SCHEMA_CONTEXT = get_prompt_context()

def generate_sql_from_nl(nl_query: str) -> str:
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SCHEMA_CONTEXT},
                {"role": "user", "content": nl_query}
            ],
            temperature=0,
        )
        sql = response.choices[0].message.content.strip()
        
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
            
        return sql.strip()
    except Exception as e:
        return f"-- LLM error: {e}"
