def run_query(con, sql: str):
    try:
        result = con.execute(sql).fetchall()
        return result
    except Exception as e:
        return f"SQL Error: {e}"
