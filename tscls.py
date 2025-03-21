import cv2
import pytesseract
import numpy as np
import os
import pandas as pd

# !!!Adjust the path to your Tesseract installation.!!!
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def get_text_and_confidences(img, text_conf_threshold=60):
    """
    Returns (text_found, combined_text_string)
    """
    # Use PyTesseract to get detailed info (confidences, text)
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    # Collect text fragments where conf > text_conf_threshold
    text_fragments = []
    for i, conf_str in enumerate(ocr_data["conf"]):
        conf = int(conf_str)
        if conf > text_conf_threshold:
            fragment = ocr_data["text"][i].strip()
            # Keep only fragments that have at least 1 character
            if len(fragment) > 0:
                text_fragments.append(fragment)
    
    combined_text = "".join(text_fragments)
    text_found = (len(text_fragments) > 0)
    return text_found, combined_text

def logo_has_text(image_path, text_conf_threshold=65, min_length=2):
    """
    If the logo has text above the threshold and meets minimum length,
    classify as "text_logo". Otherwise, "graphic_logo".
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # Light blur to reduce random noise that might look like letters
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Extract text and check confidence
    found_text, combined_text = get_text_and_confidences(img, text_conf_threshold)
    # Check the total length of recognized text to avoid noise
    if found_text and len(combined_text) >= min_length:
        return "text_logo", combined_text
    return "graphic_logo", None

def tscls(logo_dir):
    results = []

    for filename in os.listdir(logo_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(logo_dir, filename)
            result = logo_has_text(image_path, text_conf_threshold=50, min_length=2)
            results.append((filename, result[0], result[1]))

    df = pd.DataFrame(results, columns=["filename", "classification", "text"])
    return df