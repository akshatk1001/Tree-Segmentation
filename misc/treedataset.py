import os
from torch.utils.data import Dataset
import numpy as np
import cv2

class TreeDataSet(Dataset):
    def __init__(self, images_dir, masks_dir, transformation = None):
        super().__init__()
        self.images_path = images_dir
        self.masks_path = masks_dir
        self.transformation = transformation
        self.images_list = os.listdir(images_dir)

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.images_path, self.images_list[index])
        mask_path = os.path.join(self.masks_path, self.images_list[index].replace(".jpg", ".png"))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32)
        mask[mask == 255.0] = 1.0

        if self.transformation is not None:
            augmented_pics = self.transformation(image = image, mask = mask)
            image = augmented_pics["image"]
            mask = augmented_pics["mask"]

        return image, mask
    