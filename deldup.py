from PIL import Image
import imagehash
import os

def get_image_hash(image_path):
    """Generates a perceptual hash for an image"""
    img = Image.open(image_path).convert("L")
    return imagehash.phash(img)  # Perceptual hash

def get_hashes(logo_dir):
    hash_dict = {}
    # Iterate through images and detect duplicates
    for filename in os.listdir(logo_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(logo_dir, filename)
            img_hash = get_image_hash(image_path)
            
            if img_hash in hash_dict:
                print(f"Duplicate found: {filename} is similar to {hash_dict[img_hash]}")
                os.remove(image_path)
            else:
                hash_dict[img_hash] = filename
    
    return hash_dict    

import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np

def load_resnet():
    resnet = models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Identity()  # Remove final layer
    resnet.eval()
    return resnet

def load_transform():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Imagenet standards
    ])
    return transform

def extract(image_path, resnet, transform):
    """Extracts feature vector from an image using ResNet50"""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        features = resnet(img_tensor)

    return features.numpy().flatten()

# Extract features for all logos
def extract_features(logo_dir, resnet, transform):
    logo_features = {}
    for filename in os.listdir(logo_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(logo_dir, filename)
            logo_features[filename] = extract(image_path, resnet, transform)
    return logo_features, np.array(list(logo_features.values()))

from sklearn.cluster import DBSCAN

# Apply DBSCAN clustering
def apply_dbscan(logo_features, feature_matrix):
    dbscan = DBSCAN(eps=10, min_samples=2, metric='euclidean')
    clusters = dbscan.fit_predict(feature_matrix)

    # Assign clusters to filenames
    cluster_dict = {}
    for i, filename in enumerate(logo_features.keys()):
        cluster_id = clusters[i]
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(filename)
    
    return cluster_dict

import regex as re
from rapidfuzz import process, fuzz

# Function to extract brand name
def extract_brand(filename):
    return re.split(r'[-_.]', filename)[0]

# Process each cluster
def process_clusters(cluster_dict, logo_dir):
    for cluster_id, images in cluster_dict.items():
        seen_brands = set()
        to_delete = []  # Track files to delete

        # Use fuzzy matching to detect similar brand names
        remaining_images = images.copy()
        while remaining_images:
            base_image = remaining_images.pop(0)
            base_brand = extract_brand(base_image)
            cluster = [base_image]

            matches = process.extract(base_brand, remaining_images, scorer=fuzz.partial_ratio)
            for match, score, index in matches:
                if score >= 85:  # Similarity threshold
                    cluster.append(match)
                    remaining_images.remove(match)

            # Keep only one logo from the cluster, delete the rest
            keep = cluster[0]
            seen_brands.add(keep)
            to_delete.extend(cluster[1:])

        # Delete
        for image in to_delete:
            image_path = os.path.join(logo_dir, image)
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted: {image}")