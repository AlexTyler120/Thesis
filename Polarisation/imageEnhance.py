import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_expansion(image):
    # Convert to float to prevent clipping values
    image = image.astype(np.float32)
    
    # Find the minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)
    
    # Perform histogram expansion
    expanded = (image - min_val) * (255 / (max_val - min_val))
    
    # Convert back to uint8
    expanded = np.clip(expanded, 0, 255).astype(np.uint8)
    
    return expanded
