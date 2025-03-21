import os
from load import download_logo, preprocess
from concurrent.futures import ThreadPoolExecutor
from deldup import get_hashes, load_resnet, load_transform, extract_features, apply_dbscan, process_clusters
from sklearn.cluster import DBSCAN
import regex as re
from rapidfuzz import process, fuzz
from rmbg import detect_background_color, resize_and_pad, crop_background, standardize_logos
from tscls import get_text_and_confidences, logo_has_text, tscls
import pytesseract
from color import get_dominant_colors, color_to_vector, feature_vector, create_dataframe, best_threshold, clustering
import pandas as pd
from cp_cls import compute_patchwise_lightness_saturation, analyze_logo_color_categories, map_patch_to_category, analyze_logo_patches, matching_the_category, cp_cls
from min_cls import min_cls
import joblib
from train_labeling_websites import download_nltk_data, df_and_csv, scrape_websites, translate_all_websites, train_NB_model
from industry import create_df, scrape_filenames, predict_labels
from resnet import load_images, split_data, create_generators, build_model, train_model, evaluate_model, generate_reports, plot_metrics, save_model, predict_image
import numpy as np
import tensorflow as tf

# Uncomment the next line if you want to run the script exclusively on CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Download logos - directory 'logos' will be created
num_threads = os.cpu_count() * 2
df, save_dir, websites = preprocess('logos.snappy.parquet')
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    executor.map(download_logo, websites)
print("Downloading complete!")

# Remove duplicates by perceptual hashes
hashes = get_hashes(save_dir)

# Load ResNet50 and transform function
resnet = load_resnet()
transform = load_transform()

# Extract features for all logos
logo_features, feature_matrix = extract_features(save_dir, resnet=resnet, transform=transform)
print("Feature extraction complete!")

# Cluster logos using DBSCAN
cluster_dict = apply_dbscan(logo_features, feature_matrix)
process_clusters(cluster_dict, save_dir)

print("Optimized duplicate removal complete!")

# Standardize logos
standardize_logos(save_dir)

### LOGO CLASSIFICATION ###

# 1) Text/Symbol Classification

# For this classification task, we will use Tesseract OCR to extract text from logos.
# If you don't have Tesseract installed, or don't want to use it, comment next 5 lines.
tesseract_exec_path = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' # Change this to your Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = tesseract_exec_path 

df_ts = tscls(save_dir)
df_ts.to_csv('logo_ts_cls.csv', index=False)
print("Text/Symbol classification complete! Check 'logo_ts_cls.csv' for results.")

# 2) Color Classification

logo_features = feature_vector(save_dir)
df, X_scaled = create_dataframe(logo_features)
threshold = best_threshold(X_scaled)
df = clustering(df, X_scaled, threshold)
print("Color classification complete! Check 'logo_color_cls.csv' for results.")
# 3) Color Palette Classification

df_cp = cp_cls(save_dir)
print("Color Palette classification complete! Check 'logo_cp_cls.csv' for results.")

# 4) Minimal Logo Classification

df_min = min_cls(save_dir)
print("Minimal Logo classification complete! Check 'logo_min_cls.csv' for results.")

# 5) Industry Classification using ResNet50

# Load the ResNet50 model and train it
# If you want to skip training and use the pre-trained model, comment the next 9 lines.
images = load_images()
train_df, test_df = split_data(images)
train_images, val_images, test_images = create_generators(train_df, test_df)
model = build_model()
history = train_model(model, train_images, val_images)
evaluate_model(model, test_images)
generate_reports(model, test_images)
plot_metrics(history)
save_model(model)

# Predict the labels for the logos
model_resnet = tf.keras.models.load_model('industry.h5')
df_resnet = create_df(save_dir)
# df_resnet['Classification'] = df_resnet['filename'].apply(lambda x: np.argmax(predict_image(model_resnet, os.path.join(save_dir, x))))
# label_df(df_resnet, model_resnet)
df_resnet = predict_image(df_resnet, model_resnet)
df_resnet.to_csv('logo_resnet.csv', index=False)
print("Industry classification using ResNet50 complete! Check 'logo_resnet.csv' for results.")

# 6) Industry Classification using Naive Bayes

download_nltk_data()
# Train the Naive Bayes model on the scraped text data
# If you want to skip training and use the pre-trained model, comment the next 8 lines.
dataset_NB = pd.read_csv('websites.csv')
texts = scrape_websites(dataset_NB)
dataset_NB = df_and_csv(dataset_NB, texts)
translated_texts = translate_all_websites(dataset_NB)
dataset_NB['text'] = translated_texts
dataset_NB = dataset_NB.dropna(subset=['text'])
model = train_NB_model(dataset_NB)
print("Naive Bayes model training complete!")

# Scrape websites and predict labels for each logo
model_text = joblib.load('text_classification_model.pkl')
df_NB = create_df(save_dir)
scraped_texts = scrape_filenames(df_NB)
df_NB['text'] = scraped_texts
translated_texts = translate_all_websites(df_NB)
df_NB['text'] = translated_texts
predict_labels(df_NB, model_text)
print("Industry classification using Naive Bayes complete! Check 'logo_NB_cls.csv' for results.")