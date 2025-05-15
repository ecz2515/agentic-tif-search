from typing import List, Dict, Any, Optional
import openai
import json
import pandas as pd
from openai.types.chat import ChatCompletionMessageParam
from db.init_duckdb import load_expenditures_table
from runner.executor import run_query
from llm.query_planner import generate_sql_from_nl
from llm.result_humanizer import humanize_query_result
from pdf_index.query_pdf import query_pdf_index
from db.schema_discovery import get_prompt_context
import time
from io import StringIO

# Get schema context for function descriptions
SCHEMA_CONTEXT = get_prompt_context()

# Define our available functions
FUNCTIONS = [
    {
        "name": "get_schema_info",
        "description": "Get information about the database schema and example queries",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "query_sql_database",
        "description": "Query the TIF expenditures database using natural language",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language query to convert to SQL and execute"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_pdf_documents",
        "description": "Search through TIF PDF documents and reports",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information in PDFs"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "humanize_result",
        "description": "Convert a technical result into a human-friendly response",
        "parameters": {
            "type": "object",
            "properties": {
                "original_query": {
                    "type": "string",
                    "description": "The user's original question"
                },
                "technical_result": {
                    "type": "string",
                    "description": "The technical result to be humanized"
                },
                "source": {
                    "type": "string",
                    "description": "The source of the data (sql or pdf)",
                    "enum": ["sql", "pdf"]
                }
            },
            "required": ["original_query", "technical_result", "source"]
        }
    }
]

class TIFAgent:
    def __init__(self, client: openai.OpenAI):
        self.client = client
        self.con = load_expenditures_table()  # Initialize database connection
        self.conversation_history: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": """You are an AI assistant specialized in helping users understand Tax Increment Financing (TIF) data.
                You can query a SQL database of TIF expenditures and search through PDF documents and reports.
                Always try to provide comprehensive answers by combining information from both sources when relevant.
                Be precise with numbers and dates, and always cite your sources.
                
                When answering questions:
                1. For numerical data, use the SQL database
                2. For goals, objectives, and qualitative information, use the PDF documents
                3. Combine both sources when a question requires both types of information
                4. Always provide context and cite your sources.
                
                IMPORTANT: Always use function calling to interact with the system. Do not try to generate SQL queries directly."""
            }
        ]

    def add_message(self, role: str, content: str, name: Optional[str] = None):
        """Add a message to the conversation history."""
        message = {"role": role, "content": content}
        if name:
            message["name"] = name
        self.conversation_history.append(message)

    def process_query(self, query: str) -> str:
        """Process a user query using function calling."""
        self.add_message("user", query)
        sources_used = set()
        start_time = time.time()
        executed_queries = set()  # Track executed queries to prevent duplicates

        while True:
            try:
                # Get the response with potential function calls
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=self.conversation_history,
                    functions=FUNCTIONS,
                    function_call="auto"
                )

                message = response.choices[0].message
                if message.content:
                    print(f"\nAssistant message: {message.content}\n")  # Only log non-empty messages
                
                # If there's no function call, we're done
                if not message.function_call:
                    # Add source attribution to the final response
                    if sources_used:
                        sources_text = " and ".join(sorted(sources_used))
                        if not message.content:
                            return "I apologize, but I couldn't generate a meaningful response. Please try rephrasing your question."
                        return f"{message.content}\n\n[Source: {sources_text}]"
                    return message.content or "I apologize, but I couldn't generate a response. Please try rephrasing your question."

                # Add the assistant's message to the conversation
                self.add_message("assistant", message.content or "")

                # Execute the function
                function_name = message.function_call.name
                function_args = message.function_call.arguments
                print(f"\nFunction call: {function_name} with args: {function_args}\n")  # Log the function call
                
                # Skip if we've already executed this exact query
                if function_name == "query_sql_database":
                    query_hash = hash(function_args)
                    if query_hash in executed_queries:
                        print(f"Skipping duplicate SQL query")
                        continue
                    executed_queries.add(query_hash)
                
                function_start_time = time.time()
                function_result = self._execute_function(function_name, function_args)
                function_time = time.time() - function_start_time
                print(f"Function {function_name} took {function_time:.2f} seconds")
                
                # Track which sources we've used
                if function_name == "query_sql_database":
                    sources_used.add("SQL Database")
                elif function_name == "search_pdf_documents":
                    sources_used.add("PDF Documents")
                
                # Add the function result to the conversation
                self.add_message("function", function_result, name=function_name)
                
            except Exception as e:
                print(f"Error in process_query: {str(e)}")
                return f"I encountered an error while processing your question: {str(e)}. Please try again."

    def _execute_function(self, function_name: str, function_args: str) -> str:
        """Execute a function and return its result."""
        args = json.loads(function_args)
        
        if function_name == "get_schema_info":
            return SCHEMA_CONTEXT
            
        elif function_name == "query_sql_database":
            try:
                # Generate SQL from natural language
                nl_query = args["query"]
                sql_query = generate_sql_from_nl(nl_query)
                print(f"\nGenerated SQL query: {sql_query}\n")
                if sql_query.startswith("--"):
                    return f"Error generating SQL: {sql_query}"
                
                # Execute the query
                result = run_query(self.con, sql_query)
                if not result or len(result) == 0:
                    return "No results found in the database."
                
                # Convert the result to a DataFrame for better handling
                if isinstance(result, str) and result.startswith("SQL Error"):
                    return result
                
                # Get column names from the SQL query
                col_names = [desc[0] for desc in self.con.execute(sql_query).description]
                df = pd.DataFrame(result, columns=col_names)
                
                # Use humanize_query_result to format the response
                return humanize_query_result(nl_query, sql_query, df)
            except Exception as e:
                return f"Error executing SQL query: {str(e)}"
                
        elif function_name == "search_pdf_documents":
            try:
                query = args["query"]
                result = query_pdf_index(query)
                return result
            except Exception as e:
                return f"Error searching PDF documents: {str(e)}"
                
        elif function_name == "humanize_result":
            try:
                original_query = args["original_query"]
                technical_result = args["technical_result"]
                source = args["source"]
                
                if source == "sql":
                    try:
                        # For SQL results, we can use the technical_result directly
                        # since it's already formatted by humanize_query_result
                        return technical_result
                    except Exception as e:
                        print(f"Error handling SQL result: {str(e)}")
                        return f"Raw SQL result: {technical_result}"
                else:
                    # For PDF results, we can return them as is since they're already human-readable
                    return technical_result
            except Exception as e:
                print(f"Error in humanize_result: {str(e)}")
                return f"Error humanizing result: {str(e)}"
                
        else:
            return f"Unknown function: {function_name}"