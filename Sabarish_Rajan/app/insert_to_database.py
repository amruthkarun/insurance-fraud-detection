import threading
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL)
table_name = os.getenv("table_name")


class DatabaseInsertion:
    def __init__(self):
        pass
    def insert_new_claims(self, new_claim_data:pd.DataFrame):
        print('Attempting to insert new record into the database.')
        try:
            new_claim_data.to_sql(
                name=table_name,
                con=engine,
                if_exists='append',
                index=False
            )
            print('Succesfully added.')
        except Exception as e:
            print(f'Encountered an error: {e}🛑')