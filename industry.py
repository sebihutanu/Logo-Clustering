import joblib
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from predict_field import preprocess_text
from googletrans import Translator
from train_labeling_websites import scrape_with_proxy
import concurrent.futures

def create_df(logo_dir):
    results = []
    for filename in os.listdir(logo_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            results.append((filename, ""))
    df = pd.DataFrame(results, columns=["filename", "classification"])
    return df

def translate_text(text):
    if not text:
        return None
    
    translator = Translator()
    try:
        translation = translator.translate(text, dest="en")
        return translation.text
    except Exception as e:
        print(f"Translation failed: {e}")
        return None

def scrape_one_filename(filename):
    url = f"https://www.{filename.removesuffix('.png')}"
    return scrape_with_proxy(url)
def scrape_filenames(df):
    filenames = df["filename"].tolist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        scraped_texts = list(executor.map(scrape_one_filename, filenames))
    return scraped_texts

def predict_labels(df, model_text):
    mask = df["text"].isna()

    # For rows where text is present:
    df.loc[~mask, "classification"] = model_text.predict(df.loc[~mask, "text"])

    # For rows where text is None:
    df.loc[mask, "classification"] = "to be classified"

    df.to_csv("logo_NB_cls.csv", columns=['filename', 'classification'], index=False)
    return df