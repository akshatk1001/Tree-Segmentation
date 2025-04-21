'''
This script is used to test the model on a single image. 
'''

from utils import test_model

test_model(model_loc="../../../data2/akhand04/DeepGIS/UNet/final_tree_segmenting_UNET.pth.tar", 
           image_loc="data/testing_images/000286.jpg", 
           image_width = 3024/6,
           image_height = 4032/6, 
           device="cuda")