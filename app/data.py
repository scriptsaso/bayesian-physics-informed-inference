
#app/data.py
import pandas as pd

def load_data(path: str):
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print("Data loaded successfully")
    return df
