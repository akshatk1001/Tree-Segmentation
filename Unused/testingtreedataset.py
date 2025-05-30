import os
from torch.utils.data import Dataset
import torch
import torchvision.transforms.v2 as transforms
import numpy as np
from PIL import Image

class TreeDataSet2(Dataset):
    def __init__(self, images_dir, masks_dir, transformation=None):
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

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask).astype(np.float32)
        mask[mask == 255.0] = 1.0
        mask = Image.fromarray(mask)

        if self.transformation is not None:
            augmented = self.transformation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # If using torchvision transforms, ensure mask is also converted appropriately
        if isinstance(self.transformation, transforms.Compose):
            mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float()

        return image, mask