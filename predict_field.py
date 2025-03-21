import requests
from bs4 import BeautifulSoup
import pandas as pd
import concurrent.futures
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def scrape_with_proxy(url, proxy="107.181.187.195:11403"):
    """
    Example function showing how to scrape a website using a given HTTP proxy.
    """
    proxy_url = f"https://hutanu2003:fwuvojgjkx@{proxy}"

    proxies = {
        "http": proxy_url,
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }

    try:
        response = requests.get(url, proxies=proxies, headers=headers, timeout=10)
        response.raise_for_status()

        # parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        print(f"Scraped {url}, text length: {len(text)}")
        return text
    except Exception as e:
        print(f"Error scraping {url} via proxy {proxy}: {e}")
        return None


def preprocess_text(text):
    if text is None:
        return None
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text[:4500]
    return text