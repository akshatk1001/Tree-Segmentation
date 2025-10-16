import os
import numpy as np
from PIL import Image

def calc_IoU(predicted_mask: np.ndarray, true_mask: np.ndarray):
    intersection = np.logical_and(predicted_mask, true_mask)
    union = np.logical_or(predicted_mask, true_mask)

    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou

def calc_dice_score(predicted_mask: np.ndarray, true_mask: np.ndarray):
    intersection = np.logical_and(predicted_mask, true_mask)
    total_pixels = np.sum(predicted_mask) + np.sum(true_mask)

    dice = 2 * np.sum(intersection) / total_pixels if np.sum(total_pixels) > 0 else 0
    return dice

def calc_accuracy(predicted_mask: np.ndarray, true_mask: np.ndarray):
    correct_pixels = np.sum(predicted_mask == true_mask)
    total_pixels = true_mask.size

    accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
    return accuracy

def calc_metrics(predicted_masks_folder : str, true_masks_folder: str):
    avg_iou = 0.0
    avg_dice = 0.0
    avg_accuracy = 0.0

    images = os.listdir(predicted_masks_folder)
    num_images = len(images)

    for filename in images:
        true_mask_path = os.path.join(true_masks_folder, filename.replace("_noiseremoved_segmented.png", "_mask.png"))
        predicted_masks_path = os.path.join(predicted_masks_folder, filename)

        # Load predicted mask
        predicted = Image.open(predicted_masks_path)
        predicted_np = np.array(predicted)
        
        # Load true mask
        true = Image.open(true_mask_path)
        true_np = np.array(true)
        
        # Convert predicted mask to binary (0s and 1s)
        predicted_binary = (predicted_np > 127).astype(np.uint8)
        
        # Convert true mask to binary and resize to match predicted mask
        if len(true_np.shape) == 3:  # RGB image
            # Convert to grayscale by taking first channel or using a threshold
            true_gray = true_np[:, :, 0] if true_np.shape[2] > 0 else true_np
        else:
            true_gray = true_np
            
        # Resize true mask to match predicted mask dimensions
        true_resized = Image.fromarray(true_gray).resize((predicted_np.shape[1], predicted_np.shape[0]), Image.Resampling.NEAREST)
        true_resized_np = np.array(true_resized)
        
        # Convert to binary (assuming non-zero pixels are foreground)
        true_binary = (true_resized_np > 0).astype(np.uint8)

        iou = calc_IoU(predicted_binary, true_binary)
        dice = calc_dice_score(predicted_binary, true_binary)
        accuracy = calc_accuracy(predicted_binary, true_binary)

        avg_iou += iou
        avg_dice += dice
        avg_accuracy += accuracy

    avg_iou /= num_images
    avg_dice /= num_images
    avg_accuracy /= num_images

    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")