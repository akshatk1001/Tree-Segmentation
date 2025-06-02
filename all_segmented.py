from utils import create_predictions
from accuracy_metrics import calc_metrics
import torch

def predictions():
    model_loc = "/code/akhand04/Tree-Segmentation/final_tree_segmenting_UNET.pth"
    images_dir = "/code/akhand04/Tree-Segmentation/data/all_images"
    image_width = int(3024 / 6)
    image_height = int(4032 / 6)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" 

    create_predictions(model_loc=model_loc, 
                       image_dir = images_dir,
                       output_dir="/code/akhand04/Tree-Segmentation/data/predicted_masks",
                       image_height=image_height,
                       image_width=image_width,
                       device=device)

def metrics():
    predicted_masks_folder = "/code/akhand04/Tree-Segmentation/data/predicted_masks/noiseremoved"
    true_masks_folder = "/code/akhand04/Tree-Segmentation/data/all_masks"

    calc_metrics(predicted_masks_folder=predicted_masks_folder, 
                 true_masks_folder=true_masks_folder)
    
if __name__ == "__main__":
    # predictions()
    metrics()
