'''
This script is used to test the model on a single image. 
'''

from utils import test_model

# loc = input("Enter the model location: ")
# if not loc:
#     loc = "/code/akhand04/Tree-Segmentation/final_tree_segmenting_UNET.pth"\
    
loc = "/code/akhand04/Tree-Segmentation/final_tree_segmenting_UNET.pth"

# image_loc = input("Enter the image location: ")
image_loc = "/code/akhand04/Tree-Segmentation/data/all_images/000081.jpg"

test_model(model_loc=loc, 
           image_loc=image_loc, 
           image_width = int(3024/6),
           image_height = int(4032/6), 
           device="cuda")