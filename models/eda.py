
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    print("Cleaning DataFrame...")
    print("Printing Head of DataFrame:")
    print(df.head())
    
    print("Printing Info of DataFrame:")
    print(df.info())
    
    print("Printing Sum of Null Values in DataFrame:")
    print(df.isnull().sum())
    
    print("Loaded DataFrame with shape:", df.shape)



