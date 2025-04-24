import duckdb
import pandas as pd

def load_expenditures_table(csv_path="data/tif_expenditures.csv"):
    df = pd.read_csv(csv_path)
    con = duckdb.connect("tif.duckdb")
    con.register("raw_df", df)
    con.execute("CREATE OR REPLACE TABLE expenditures AS SELECT * FROM raw_df")
    return con
