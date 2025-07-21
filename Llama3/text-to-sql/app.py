import streamlit as st
import ollama
from sqlalchemy import create_engine, text
import json
from db_schema import fetch_db_schema

# Set up database connection
engine = create_engine('postgresql://postgres:deadpool@localhost:5432/hogwarts')
print(f"Engine created: {engine}")

# Function to convert text to SQL using Ollama
def text_to_sql(natural_language_text):
    # table_schemas = """
    # house_points(house_name TEXT PRIMARY KEY, points INTEGER)
    # """
    table_schemas = fetch_db_schema(engine)
    
    prompt_template = f"""
    You are a SQL expert.
    
    Please help to convert the following natural language command into a valid UPDATE SQL query. Your response should ONLY be based on the given context and follow the response guidelines and format instructions.

    ===Tables
    {table_schemas}

    ===Response Guidelines
    1. If the provided context is sufficient, please generate a valid query WITHOUT any explanations for the question.
    2. Please format the query before responding.
    3. Please always respond with a valid well-formed JSON object with the following format
    4. There are only UPDATE queries and points are either added or deducted from a house.

    ===Response Format
    {{
        "query": "A valid UPDATE SQL query when context is sufficient."
    }}

    ===Command
    {natural_language_text}
    """
    
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt_template}]
    )
    
    response_content = response['message']['content']

    print(response_content)
    
    try:
        response_json = json.loads(response_content)
        if "query" in response_json:
            return response_json["query"]
        else:
            return f"Error: {response_json.get('explanation', 'No explanation provided.')}"
    except json.JSONDecodeError:
        return "Error: Failed to parse response as JSON."

# Function to execute SQL queries
def run_sql_query(query):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            conn.commit()
            return result
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return None

# Streamlit App
st.title("SQL Query Executor")

def display_response(query):
    sql_query = text_to_sql(query)
    print("Generated SQL Query:", sql_query)
    
    st.text_area("Generated SQL Query", sql_query, height=100)
    
    if st.button("Run Query"):
        result = run_sql_query(sql_query)
        st.success("Query executed successfully!") if result else st.error("Query execution failed.")

# User input
user_query = st.text_input("Enter your query:")
if user_query:
    display_response(user_query)
