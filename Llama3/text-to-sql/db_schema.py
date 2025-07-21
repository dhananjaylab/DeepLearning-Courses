from sqlalchemy.sql import text

def fetch_db_schema(engine):
    query = """
        WITH first_three_tables AS (
            SELECT t.table_name
            FROM information_schema.tables t
            WHERE t.table_schema = 'public'
            ORDER BY t.table_name
            LIMIT 3
        )
        SELECT
            t.table_name,
            c.column_name,
            c.data_type,
            c.is_nullable,
            t.table_type,
            t.table_catalog,
            tc.constraint_type,
            tc.constraint_name
        FROM
            information_schema.tables t
        LEFT JOIN
            information_schema.columns c 
            ON t.table_name = c.table_name AND t.table_schema = c.table_schema
        LEFT JOIN
            information_schema.table_constraints tc 
            ON t.table_name = tc.table_name AND t.table_schema = tc.table_schema
        WHERE
            t.table_schema = 'public'
            AND t.table_name IN (SELECT table_name FROM first_three_tables)
        ORDER BY
            t.table_name, c.ordinal_position;
    """

    def run_sql_query(query):
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                return result.fetchall()
        except Exception as e:
            return f"Error executing query: {e}"

    rows = run_sql_query(query)
    if not rows or isinstance(rows, str):  # If there's an error, return it
        return rows if isinstance(rows, str) else "No schema information found."

    table_info = {}

    for (
        table_name,
        column_name,
        data_type,
        is_nullable,
        table_type,
        table_catalog,
        constraint_type,
        constraint_name,
    ) in rows:
        if table_name not in table_info:
            table_info[table_name] = {
                "DatabaseName": table_catalog,
                "Name": table_name,
                "Description": "",
                "PartitionKeys": "None",
                "StorageDescriptor": {"Columns": []},
                "Constraints": [],
            }

        table_info[table_name]["StorageDescriptor"]["Columns"].append(
            f"- {column_name} ({data_type}, Nullable: {is_nullable})"
        )

        if constraint_type:
            table_info[table_name]["Constraints"].append(f"{constraint_type}: {constraint_name}")

    # Formatting the response as a readable string
    schema_text = "\n".join(
        [
            f"Database: {table['DatabaseName']}\n"
            f"Table: {table['Name']}\n"
            f"Description: {table['Description']}\n"
            f"Partition Keys: {table['PartitionKeys']}\n"
            f"Columns:\n" + "\n".join(table["StorageDescriptor"]["Columns"]) + "\n"
            f"Constraints:\n" + ("\n".join(table["Constraints"]) if table["Constraints"] else "None") + "\n"
            "------------------------------------"
            for table in table_info.values()
        ]
    )
    print(schema_text)
    return schema_text
