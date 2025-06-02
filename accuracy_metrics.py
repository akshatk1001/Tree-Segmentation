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
        true_mask_path = os.path.join(true_masks_folder, filename.replace("_predicted.png", "_mask.png"))
        predicted_masks_path = os.path.join(predicted_masks_folder, filename)

        predicted = Image.open(predicted_masks_path)
        true = Image.open(true_mask_path)
        predicted_np = np.array(predicted)
        true_np = np.array(true)

        iou = calc_IoU(predicted_np, true_np)
        dice = calc_dice_score(predicted_np, true_np)
        accuracy = calc_accuracy(predicted_np, true_np)

        avg_iou += iou
        avg_dice += dice
        avg_accuracy += accuracy

    avg_iou /= num_images
    avg_dice /= num_images
    avg_accuracy /= num_images

    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")