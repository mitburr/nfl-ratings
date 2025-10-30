"""Database manager for NFL stats."""
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
import pandas as pd
from config.database import DB_CONFIG, get_connection_string


class DatabaseManager:
    """Manage database connections and operations."""
    
    def __init__(self):
        self.config = DB_CONFIG
        self.engine = create_engine(get_connection_string())
    
    def get_connection(self):
        """Get a raw psycopg2 connection."""
        return psycopg2.connect(**self.config)
    
    def execute_sql_file(self, filepath):
        """Execute SQL commands from a file."""
        with open(filepath, 'r') as f:
            sql = f.read()
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        print(f"Executed SQL file: {filepath}")
    
    def insert_dataframe(self, df, table_name, if_exists='append'):
        """
        Insert a pandas DataFrame into a table.
        
        Args:
            df: pandas DataFrame
            table_name: name of the table
            if_exists: 'fail', 'replace', or 'append'
        """
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
        print(f"Inserted {len(df)} rows into {table_name}")
    
    def query_to_dataframe(self, query):
        """Execute a query and return results as DataFrame."""
        return pd.read_sql(query, self.engine)
    
    def bulk_insert(self, table_name, columns, data):
        """
        Bulk insert data efficiently.
        
        Args:
            table_name: name of the table
            columns: list of column names
            data: list of tuples with values
        """
        cols = ', '.join(columns)
        query = f"INSERT INTO {table_name} ({cols}) VALUES %s ON CONFLICT DO NOTHING"
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query, data)
            conn.commit()
        print(f"Bulk inserted {len(data)} rows into {table_name}")
    
    def test_connection(self):
        """Test database connection."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()
                    print(f"Connected to PostgreSQL: {version[0]}")
                    return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False


if __name__ == "__main__":
    # Test the connection
    db = DatabaseManager()
    db.test_connection()
