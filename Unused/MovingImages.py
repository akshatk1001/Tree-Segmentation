'''
This script was used to move images from the VOC2012 dataset to a new directory. It is also used to convert the masks to black and white.
It's no longer important and will not work since file locations are different now. It's kept for reference.
'''

import os
import cv2
from tqdm import tqdm

jpeg_directory = "data/VOC2012/JPEGImages"
segmented_directory = "data/VOC2012/SegmentationClass"

training_img_directory = "data/training_kaggle_images"
train_mask_directory = "data/training_kaggle_masks"
val_img_directory = "data/val_images"
val_mask_directory = "data/val_masks"

def move_images(image_directory, target_directory, start_num = 0, number_images = 1):
    list_of_images = os.listdir(image_directory)
    for image in tqdm(list_of_images, desc="Moving images"):
        if image.endswith('.jpg') or image.endswith('.png'):
            par_index = image.rfind("(")
            image_num = int(image[(par_index + 1):-5])
            if image_num in range(start_num, number_images + 1):
                image_path = os.path.join(image_directory, image)
                target_path = os.path.join(target_directory, image)
                os.rename(image_path, target_path)

def make_b_w(image_directory, target_directory):
    images_list = os.listdir(image_directory)
    for image in tqdm(images_list):
        specific_loc = os.path.join(image_directory, image)
        new_image = cv2.imread(specific_loc, cv2.IMREAD_GRAYSCALE)
        new_image[new_image > 0] = 255
        return_loc = os.path.join(target_directory, image)
        cv2.imwrite(return_loc, new_image)

if __name__ == "__main__":
    move_images(jpeg_directory, training_img_directory, 110)
    move_images(segmented_directory, train_mask_directory, 110)
    move_images(jpeg_directory, val_img_directory, 111, 140)
    move_images(segmented_directory, val_mask_directory, 111, 140)
    make_b_w(train_mask_directory, train_mask_directory)
    make_b_w(val_mask_directory, val_mask_directory)
