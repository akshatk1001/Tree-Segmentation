import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from UNet import UNet
from treedataset import TreeDataSet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
from tqdm import tqdm

def train_with_amp(data_loader, model, optimizer, loss_fn, device, scaler):
    model.train()
    start_time = time.time()
    for images, masks in tqdm(data_loader, desc="Training with AMP"):
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = loss_fn(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    end_time = time.time()
    epoch_time = end_time - start_time
    return epoch_time

def train_without_amp(data_loader, model, optimizer, loss_fn, device):
    model.train()
    start_time = time.time()
    for images, masks in tqdm(data_loader, desc="Training without AMP"):
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        loss.backward()
        optimizer.step()
    end_time = time.time()
    epoch_time = end_time - start_time
    return epoch_time

def measure_mixed_precision_efficiency():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMAGE_WIDTH = 3024 / 12
    IMAGE_HEIGHT = 4032 / 12
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    PIN_MEMORY = True
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 1  # For measurement, use 1 epoch

    # Define Albumentations Transform
    transform_train = A.Compose([
        A.Resize(height=int(IMAGE_HEIGHT), width=int(IMAGE_WIDTH)),
        A.Rotate(limit=35, p=0.3),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    # Create DataLoader
    train_dataset = TreeDataSet("data/training_images", "data/training_masks", transform_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True
    )

    # Baseline Training (Without AMP)
    print("Starting Baseline Training (No AMP)...")
    model_baseline = UNet(input_channels=3, output_channels=1).to(DEVICE)
    optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    epoch_time_baseline = train_without_amp(train_loader, model_baseline, optimizer_baseline, loss_fn, DEVICE)
    print(f"Baseline Training Time: {epoch_time_baseline:.2f} seconds")

    # Mixed-Precision Training
    print("\nStarting Mixed-Precision Training (AMP)...")
    model_amp = UNet(input_channels=3, output_channels=1).to(DEVICE)
    optimizer_amp = optim.Adam(model_amp.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    epoch_time_amp = train_with_amp(train_loader, model_amp, optimizer_amp, loss_fn, DEVICE, scaler)
    print(f"Mixed-Precision Training Time: {epoch_time_amp:.2f} seconds")

    # Calculate Improvements
    time_improvement = ((epoch_time_baseline - epoch_time_amp) / epoch_time_baseline) * 100
    print(f"\nTraining speed improved by {time_improvement:.2f}% using mixed-precision.")

    # Measure GPU Memory Usage (Peak Memory)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    with torch.no_grad():
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.unsqueeze(1).to(DEVICE)
            outputs = model_amp(images)
    peak_memory = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)  # in GB
    print(f"Peak GPU Memory Usage: {peak_memory:.2f} GB")

    # Optionally, save these metrics
    with open("mixed_precision_metrics.txt", "w") as f:
        f.write(f"Baseline Training Time: {epoch_time_baseline:.2f} seconds\n")
        f.write(f"Mixed-Precision Training Time: {epoch_time_amp:.2f} seconds\n")
        f.write(f"Training speed improved by {time_improvement:.2f}% using mixed-precision.\n")
        f.write(f"Peak GPU Memory Usage: {peak_memory:.2f} GB\n")

if __name__ == "__main__":
    measure_mixed_precision_efficiency()