import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from predict_field import scrape_with_proxy, preprocess_text
from googletrans import Translator
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import requests
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import learning_curve
import concurrent.futures

def scrape_with_proxy(url, proxy="107.181.187.195:11403"):
    """
    Scrape a website using a given HTTP proxy.
    """
    # This proxy should be valid still, there's also a script for testing it `testproxy.py`
    proxy_url = f"https://hutanu2003:fwuvojgjkx@{proxy}"
    proxies = {"http": proxy_url,}
    
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

def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

# Load y our dataset
def translate_text(text):
    translator = Translator()
    return translator.translate(text, dest="en").text


def scrape_one_website(website):
    return scrape_with_proxy(website)
def scrape_websites(df):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        texts = list(executor.map(scrape_one_website, df['website']))
    return texts

def df_and_csv(df, texts):
    df['text'] = texts
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].apply(preprocess_text)
    df = df.dropna(subset=['text'])
    df.to_csv('text_scraped_from_dataset.csv', index=False)
    df = pd.read_csv('text_scraped_from_dataset.csv')
    df = df.dropna(subset=['text'])
    df.to_csv('text_scraped_from_dataset.csv', index=False)
    return df

def translate_if_not_none(text):
    if not text:
        return None
    translator = Translator()
    try:
        translation = translator.translate(text, dest="en")
        return translation.text
    except Exception as e:
        print(f"Translation failed: {e}")
        return None

def translate_all_websites(df):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        translated_texts = list(executor.map(translate_if_not_none, df['text']))
    return translated_texts

def train_NB_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['industry'], test_size=0.2, random_state=42)

    # Create a pipeline for text classification
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),  # Convert text to TF-IDF features
        ('clf', MultinomialNB())       # Naive Bayes classifier
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    joblib.dump(model, 'text_classification_model.pkl')
    cm_display = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

    # Extract the underlying confusion matrix array
    cm_array = cm_display.confusion_matrix

    # Plot confustion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_array, annot=True, fmt="d", cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    train_sizes, train_scores, valid_scores = learning_curve(
        model,
        df['text'], df['industry'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # Calculate mean and standard deviation for training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std  = np.std(valid_scores, axis=1)

    # Plot the learning curve
    plt.figure(figsize=(10,6))
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training Accuracy")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)
    plt.plot(train_sizes, valid_scores_mean, 'o-', label="Validation Accuracy")
    plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.2)
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.show()
    return model
