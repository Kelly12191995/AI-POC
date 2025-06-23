import teradatasql
import pandas as pd
import csv
from datetime import date, timedelta


class DataFromDatalab:
    """Main class to handle database queries."""

    # Assign credentials inside the class
    HOST = 'bi360.int.bell.ca'
    USER = 'CAROL_LI'
    PASSWORD = 'Zxbsgzale#25743'

    def __init__(self):
        """Initialize database connection"""
        self.con = teradatasql.connect(host=self.HOST, user=self.USER, password=self.PASSWORD)
        self.START_DT = '2024-01-01'
        self.END_DT = date.today() - timedelta(days=1)

    def load_data(self, query: str) -> pd.DataFrame:
        """Generic method to fetch data based on query."""
        try:
            df = pd.read_sql(query, self.con)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def registration(self) -> pd.DataFrame:
        """Fetch Registration data"""
        query = "SELECT * FROM DL_BCO.CY_REGISTRATION_TOKENIZED"
        return self.load_data(query)

    def DEI_bell(self) -> pd.DataFrame:
        """Fetch DEI Bell data"""
        query = "SELECT * FROM DL_BCO.CL_EASY_INDEX_FLOW"
        return self.load_data(query)

    def close_connection(self):
        """Close the database connection"""
        if self.con:
            self.con.close()
            print("Database connection closed.")

# # Usage Example:
# db = DataFromDatalab()
# df = db.registration()
# df = db.DEI_bell()

# # Close connection when done
# db.close_connection()
# print(df.dtypes)
