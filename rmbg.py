import cv2
import numpy as np
import os

def detect_background_color(image, margin=0.1):
    """
    Detects if the image background is predominantly white or black by
    sampling pixels along the borders.
    """
    h, w = image.shape[:2]
    margin_h = int(margin * h)
    margin_w = int(margin * w)
    
    # Extract border regions
    top = image[0:margin_h, :]
    bottom = image[-margin_h:, :]
    left = image[:, 0:margin_w]
    right = image[:, -margin_w:]
    
    # Combine all border pixels
    border_pixels = np.concatenate((top.reshape(-1, 3),
                                    bottom.reshape(-1, 3),
                                    left.reshape(-1, 3),
                                    right.reshape(-1, 3)), axis=0)
    
    # Compute the average intensity across all channels
    avg_intensity = np.mean(border_pixels)
    
    if avg_intensity > 200:
        return "white"
    elif avg_intensity < 55:
        return "black"
    else:
        return "unknown"

def crop_background(input_path):
    """
    Detect background color (white or black) and crop the image accordingly.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")

    bg_color = detect_background_color(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if bg_color == "white":
        threshold_value = 240
        # White background => pixels above threshold are background
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        mask_inverted = 255 - mask  # we want the logo to be 255
    elif bg_color == "black":
        threshold_value = 30
        # Black background => pixels below threshold are background
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        mask_inverted = 255 - mask
    else:
        return img

    coords = cv2.findNonZero(mask_inverted)
    if coords is None:
        # Entire image is background
        return img

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]

    cv2.imwrite(input_path, cropped)

    return cropped

def resize_and_pad(image, target_size=256, border_color=(255, 255, 255)):
    """
    Resizes an image so its longest side equals target_size and adds padding 
    to make it square.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=border_color)
    return padded

def standardize_logos(input_dir):
    """
    Standardizes logos by cropping and resizing them.
    """
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            img = cv2.imread(input_path)
            bg_color_detected = detect_background_color(img)
            
            # If uncertain, consider defaulting to white.
            if bg_color_detected == "unknown":
                bg_color_detected = "white"

            cropped_img = crop_background(input_path)

            # Set border color for padding
            if bg_color_detected == "black":
                pad_color = (0, 0, 0)
            else:
                pad_color = (255, 255, 255)
            
            # Standardize image size by resizing and padding
            standardized_img = resize_and_pad(cropped_img, target_size=256, border_color=pad_color)
            output_standardized_path = os.path.join(input_dir, filename)
            cv2.imwrite(output_standardized_path, standardized_img)
