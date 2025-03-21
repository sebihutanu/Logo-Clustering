import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def compute_patchwise_lightness_saturation(img_path):
    """
    Loads a 256x256 image, converts to HLS,
    splits into 1024 patches (8x8 each),
    and returns a list of (L, S) means for each patch.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image from {img_path}")
    
    img_bgr = cv2.resize(img_bgr, (256, 256))
    
    # Convert to HLS (Hue, Lightness, Saturation)
    img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    H, L, S = cv2.split(img_hls)
    
    # Create 8x8 patches => total 32x32 = 1024 patches.
    patch_size = 8
    patchwise_values = []  # (meanL, meanS) for each patch
    
    for row_block in range(32):
        for col_block in range(32):
            r_start = row_block * patch_size
            r_end   = r_start + patch_size
            c_start = col_block * patch_size
            c_end   = c_start + patch_size
            block_L = L[r_start:r_end, c_start:c_end]
            block_S = S[r_start:r_end, c_start:c_end]
            mean_L = np.mean(block_L)
            mean_S = np.mean(block_S)
            patchwise_values.append((mean_L, mean_S))
    
    return patchwise_values

def analyze_logo_patches(img_path, show=False):
    # Get the (L, S) pairs for each patch
    patchwise_ls = compute_patchwise_lightness_saturation(img_path)
    patchwise_ls = np.array(patchwise_ls)
    
    df = pd.DataFrame(patchwise_ls, columns=["Lightness", "Saturation"])
    # Normalize to [0, 1]
    df["Lightness"] = df["Lightness"] / 255.0
    df["Saturation"] = df["Saturation"] / 255.0
    if show:
        L_vals = df["Lightness"].values
        S_vals = df["Saturation"].values
        plt.figure(figsize=(6, 6))
        plt.scatter(S_vals, L_vals, alpha=0.5, s=10, c='blue')
        plt.xlabel("Saturation (0-1)")
        plt.ylabel("Lightness (0-1)")
        plt.grid(True)
        plt.show()
    
    return df

def map_patch_to_category(L, S):
    # Returns a category based on Lightness and Saturation values
    if L < 0.3:
        if S < 0.8:
            return "dark"
        else:
            return "deep"
    elif L < 0.66:
        if S < 0.66:
            return "deep"
        else:
            return "colorblind"
    else:
        if S < 0.33:
            return "pastel"
        elif S < 0.66:
            return "muted"
        else:
            return "bright"


def analyze_logo_color_categories(df):
    categories_count = {
        "dark": 0,
        "deep": 0,
        "colorblind": 0,
        "pastel": 0,
        "muted": 0,
        "bright": 0
    }
    for i, row in df.iterrows():
        L, S = row["Lightness"], row["Saturation"]
        category = map_patch_to_category(L, S)
        categories_count[category] += 1
    for category in categories_count:
        categories_count[category] /= 1024
    return categories_count


def matching_the_category(categories_count):
    """
    possible categories:
    dark, deep, colorblind, pastel, muted, bright
    dark-deep, dark-colorblind, dark-pastel, dark-muted, dark-bright
    deep-colorblind, deep-pastel, deep-muted, deep-bright
    colorblind-pastel, colorblind-muted, colorblind-bright
    pastel-muted, pastel-bright
    muted-bright
    """
    result_posibilities = ["dark", "deep", "colorblind", "pastel", "muted", "bright",
                            "dark-deep", "dark-colorblind", "dark-pastel", "dark-muted", "dark-bright",
                            "deep-colorblind", "deep-pastel", "deep-muted", "deep-bright",
                            "colorblind-pastel", "colorblind-muted", "colorblind-bright",
                            "pastel-muted", "pastel-bright",
                            "muted-bright"]
    max_category = max(categories_count, key=categories_count.get)
    if categories_count[max_category] > 0.45:
        return max_category
    else:
        categories_count.pop(max_category)
        second_max_category = max(categories_count, key=categories_count.get)
        result = max_category + "-" + second_max_category
        if result not in result_posibilities:
            result = second_max_category + "-" + max_category
        return result

def cp_cls(logo_dir):
    results = []
    for filename in os.listdir(logo_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(logo_dir, filename)
            patches_df = analyze_logo_patches(image_path)
            categories = analyze_logo_color_categories(patches_df)
            classification = matching_the_category(categories)
            results.append((filename, classification))
    df = pd.DataFrame(results, columns=["filename", "classification"])
    df.to_csv("logo_cp_cls.csv", index=False)
    return df