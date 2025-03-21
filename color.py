import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import os
import shutil

def get_dominant_colors(image_path, k=5):
    """
    Returns the dominant colors in an image using k-means clustering.
    The image is converted to LAB color space before clustering.
    """
    # Load image and convert to LAB color space (more accurate for color clustering)
    img_bgr = cv2.imread(image_path)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    pixels = img_lab.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    k_adjusted = min(k, unique_colors.shape[0])
    kmeans = KMeans(n_clusters=k_adjusted, random_state=42, n_init=10)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    # Compute proportions
    label_counts = np.bincount(labels)
    total_count = len(labels)
    proportions = label_counts / total_count
    # Sort by proportion (DESC)
    sorted_indices = np.argsort(-proportions)
    sorted_centers = centers[sorted_indices]
    sorted_proportions = proportions[sorted_indices]

    # Return as list of (L, A, B, proportion)
    result = [(sorted_centers[i][0],
               sorted_centers[i][1],
               sorted_centers[i][2],
               sorted_proportions[i]) for i in range(sorted_indices.size)]

    return result

def color_to_vector(colors, k=5):
    """
    Converts the list of dominant colors into a fixed-length vector.
    Each dominant color is represented by (L, A, B, proportion).
    Pads with zeros if there are fewer than k colors.
    """
    vector = []
    for i in range(k):
        if i < len(colors):
            l, a, b, prop = colors[i]
            vector.extend([l, a, b, prop])
        else:
            # Pad with zeros (or another default value) if not enough colors.
            vector.extend([0, 0, 0, 0])
    return vector


def feature_vector(logo_dir):
    """
    Extracts feature vectors for each logo.
    """
    logo_features = []
    for filename in os.listdir(logo_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(logo_dir, filename)
            dom_colors = get_dominant_colors(path, k=5)
            feature_vec = color_to_vector(dom_colors, k=5)
            logo_features.append((filename, feature_vec))
    
    return logo_features

def create_dataframe(logo_features):
    df = pd.DataFrame(logo_features, columns=["filename", "feature_vec"])
    X = np.array(df["feature_vec"].tolist())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled

def best_threshold(X_scaled):
    """
    Returns the best distance threshold for Agglomerative Clustering.
    """
    distance_thresholds = [5, 10, 15, 20, 25, 30]
    best_threshold = None
    best_score = -1

    for thr in distance_thresholds:
        agg = AgglomerativeClustering(n_clusters=None, distance_threshold=thr, metric='euclidean', linkage='ward')
        labels = agg.fit_predict(X_scaled)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_threshold = thr
    return best_threshold

def clustering(df, X_scaled, best_threshold):
    """
    Clusters the logos using Agglomerative Clustering.
    """
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=best_threshold, metric='euclidean', linkage='ward')
    labels = agg.fit_predict(X_scaled)
    df["classification"] = labels
    df.to_csv("logo_color_cls.csv", index=False)
    return df
