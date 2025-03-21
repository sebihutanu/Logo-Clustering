import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor

def preprocess(file):
    df = pd.read_parquet('logos.snappy.parquet', engine='pyarrow').drop_duplicates()

    # Ensure save directory exists
    save_dir = 'logos'
    os.makedirs(save_dir, exist_ok=True)
    websites = df['domain'].drop_duplicates().dropna().tolist()
    return df, save_dir, websites

# Function to download a logo
def download_logo(website, save_dir='logos'):
    """Downloads a logo using Clearbit API and saves it."""
    if not isinstance(website, str) or website.strip() == "":
        return
    
    url = f'https://logo.clearbit.com/{website}'
    save_path = os.path.join(save_dir, f"{website}.png")
    
    try:
        response = requests.get(url, params={'size': 248, 'format': 'png'}, timeout=5)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {website}")
        else:
            print(f"Failed: {website} (Status {response.status_code})")
    except requests.RequestException as e:
        print(f"Error fetching {website}: {e}")