import cv2
import numpy as np
import os
import pandas as pd

def is_minimalist_logo(image_path, edge_ratio_threshold=0.04):
    """
    Determines whether a logo is 'minimalist' by measuring edge density.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Compute the ratio of edge pixels to total
    total_pixels = img.shape[0] * img.shape[1]
    edge_pixels = np.count_nonzero(edges)
    edge_ratio = edge_pixels / total_pixels

    is_minimalist = (edge_ratio < edge_ratio_threshold)
    return is_minimalist, edge_ratio

def min_cls(logo_dir):
    threshold = 0.04  # 4% edge ratio
    results = []

    for filename in os.listdir(logo_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(logo_dir, filename)
            minimalist, ratio = is_minimalist_logo(image_path, edge_ratio_threshold=threshold)
            results.append((filename, minimalist, ratio))
    
    df = pd.DataFrame(results, columns=["filename", "is_minimalist", "edge_ratio"])
    df.to_csv("logo_min_cls.csv", index=False)
    return df
