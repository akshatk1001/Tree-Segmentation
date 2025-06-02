import cv2
import numpy as np
import torch
from PIL import Image
import os

"""FINAL CHOICE: remove_noise_connected_components. This had the best results for the dataset."""

def remove_noise_morphological(mask, kernel_size=5, min_area=100):
    """
    Remove noise using morphological operations
    
    Args:
        mask: Binary mask (numpy array)
        kernel_size: Size of morphological kernel
        min_area: Minimum area to keep connected components
    """
    # Convert to binary if needed
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    # Threshold to ensure binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Opening (erosion followed by dilation) - removes small noise
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Closing (dilation followed by erosion) - fills small holes
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def remove_noise_connected_components(mask, min_area=100):
    """
    Remove small connected components based on area
    """
    if mask.ndim == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create output image
    cleaned = np.zeros_like(binary)
    
    # Keep only components with area >= min_area
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
    
    return cleaned

def fill_nearby_pixels(mask, kernel_size=5):
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    filled = cv2.dilate(binary, kernel, iterations=1)
    return filled

def remove_noise_advanced(mask, morph_kernel=5, min_area=200, median_kernel=3):
    """
    Combine multiple noise removal techniques for best results
    """

    # Step 1: Connected components
    step1 = remove_noise_connected_components(mask, min_area)

    # Step 2: Morphological operations
    step2 = remove_noise_morphological(step1, morph_kernel)

    return step2

# Example usage
if __name__ == "__main__":
    # Test with a single image
    input_path = "/code/akhand04/Tree-Segmentation/data/segmented_images/141/prediction_0_epoch_141.png"
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # Try different methods
    cleaned_advanced = remove_noise_advanced(img)
    cleaned_morphological = remove_noise_morphological(img)
    cleaned_connected = remove_noise_connected_components(img)

    # Save result
    cv2.imwrite("connected_mask_141_01.png", cleaned_connected)
    cv2.imwrite("morphological_mask_141_01.png", cleaned_morphological)
    cv2.imwrite("advanced_mask_141_01.png", cleaned_advanced)

