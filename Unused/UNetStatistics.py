import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from UNet import UNet
import torch.optim as optim
from treedataset import TreeDataSet
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
import time

LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
IMAGE_WIDTH = 3024/6
TRAIN_IMG_DIR = "data/training_images"
TRAIN_MASK_DIR = "data/training_masks"
IMAGE_HEIGHT = 4032/6
NUM_WORKERS = 8
VAL_IMG_DIR = "data/val_images"
VAL_MASK_DIR = "data/val_masks"
MODEL_LOC = "../../../data2/akhand04/DeepGIS/UNet/final_tree_segmenting_UNET.pth"

'''Albumentations Pipeline Time'''

def albumentation_transform(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Rotate(limit=35, p=0.35),
        A.Normalize(
            mean = [0.0, 0.0, 0.0],
            std = [1.0, 1.0, 1.0],
            max_pixel_value = 255.0
        ),
        ToTensorV2()
    ])

def albumentation_val_transform(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(
            mean = [0.0, 0.0, 0.0],
            std = [1.0, 1.0, 1.0],
            max_pixel_value = 255.0
        ),
        ToTensorV2()
    ])


train_dataset = TreeDataSet(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transformation = albumentation_transform(IMAGE_HEIGHT, IMAGE_WIDTH))
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle = True)

val_dataset = TreeDataSet(VAL_IMG_DIR, VAL_MASK_DIR, transformation = albumentation_val_transform(IMAGE_HEIGHT, IMAGE_WIDTH))
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle = False)

model = torch.load(MODEL_LOC)
model = model.to(DEVICE) 
model.eval()  


def calculate_accuracy_iou(model, device, threshold = 0.5):
    model.eval()
    total_pixels = 0
    correct_pixels = 0
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Calculating Metrics"):
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)

            outputs = torch.sigmoid(model(images))
            preds = (outputs > threshold).float()

            # Accuracy
            correct = (preds == masks).sum().item()
            total = masks.numel()
            correct_pixels += correct
            total_pixels += total

            # IoU
            intersection = (preds * masks).sum(dim=(2, 3))
            union = (preds + masks - preds * masks).sum(dim=(2, 3))
            iou = (intersection + 1e-6) / (union + 1e-6)
            total_iou += iou.sum().item()
            num_batches += masks.size(0)
    
    accuracy = (correct_pixels / total_pixels) * 100
    average_iou = (total_iou / num_batches) * 100

    return accuracy, average_iou

accuracy, iou = calculate_accuracy_iou(model, DEVICE)
print(f"Validation Accuracy: {accuracy:.2f}%")
print(f"Validation IoU: {iou:.2f}%")





def estimate_annotation_time_reduction():
    # Estimated times (in seconds)
    manual_time_per_image = 90
    model_inference_time = 1  # 1 second per image
    adjustment_time_per_image = 25  # 25s per image

    total_manual_time = manual_time_per_image
    total_model_time = model_inference_time + adjustment_time_per_image

    # Calculate reduction
    time_reduction = ((total_manual_time - total_model_time) / total_manual_time) * 100
    print(f"Manual annotation time reduced by approximately {time_reduction:.2f}%.")
    