import torch # import pytorch
from treedataset import TreeDataSet # import the TreeDataSet class from treedataset2.py
from torch.utils.data import DataLoader # import DataLoader from torch.utils.data to create data loaders
from PIL import Image # import Image from PIL to work with images
from torchvision import transforms # import transforms from torchvision to transform images to tensors and apply other transformations
from torchvision.utils import save_image  # import save_image from torchvision.utils to save images/results
import os # import os to work with directories
import random # import random to randomly select images from the directory
import shutil # import shutil to copy files from one directory to another
import numpy as np # import numpy to work with arrays
import albumentations as A # import Albumentations to apply image augmentations
from albumentations.pytorch import ToTensorV2 # import ToTensorV2 from albumentations.pytorch to convert images to tensors
from remove_noise import remove_noise_connected_components # import the remove_noise_connected_components function from remove_noise.py to remove noise from the masks
import cv2 # import OpenCV to work with images

def create_loader(train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, batch_size, num_workers, train_transform, val_transform, pin_memory): 
    '''
    This function creates the train and validation data loaders using the TreeDataSet class.

    train_images_dir: directory containing training images
    train_masks_dir: directory containing training masks
    val_images_dir: directory containing validation images
    val_masks_dir: directory containing validation masks
    batch_size: batch size for training (how many images to process at once) - a number between 1 and the number of images
    num_workers: number of workers to use for loading data (usually set to 2*number of cores) - a number
    train_transform: transformations to apply to training images - using Albumentations image augmentation library
    val_transform: transformations to apply to validation images - also using Albumentations image augmentation, different from training transformations since we aren't rotating val images
    pin_memory: whether to copy tensors to CUDA pinned memory - this is useful for faster data transfer to GPU. Either True or False.
    '''
    train_dataset = TreeDataSet(train_images_dir, train_masks_dir, train_transform) # create training dataset using TreeDataSet class and train directory/transformation
    val_dataset = TreeDataSet(val_images_dir, val_masks_dir, val_transform) # create validation dataset using TreeDataSet class and val directory/transformation

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle = True) # create training data loader using DataLoader
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle = True) # create validation data loader using DataLoader

    return train_loader, val_loader # return the training and validation data loaders


def create_train_dataset(train_images_dir, train_masks_dir, batch_size, num_workers, train_transform, pin_memory):
    train_dataset = TreeDataSet(train_images_dir, train_masks_dir, train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle = True)

    return train_loader

def create_val_dataset(val_images_dir, val_masks_dir, batch_size, num_workers, val_transform, pin_memory):
    val_dataset = TreeDataSet(val_images_dir, val_masks_dir, val_transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle = True)
    return val_loader 

def validate(data_loader, model, loss_fn, device):
    '''
    This function validates the model on the validation data loader. It calculates the loss on the validation set.

    data_loader: validation data loader (likely the one we created in create_loader)
    model: the model we are validating (UNet in this case)
    loss_fn: the loss function we are using (BCEWithLogitsLoss in this case)
    device: the device we are using (CPU (worst case) or CUDA (NVIDIA) or MPS (macOS))
    '''
    model.eval() # set the model to evaluation mode so that it doesn't update weights
    total_loss = 0 # initialize the total loss to 0
    
    with torch.no_grad(): # disable gradient computation for validation
        for image, mask in data_loader: # iterate over the data loader (validation set) which contains images and masks in batches
            image = image.to(device) # move the image to the device (CPU or CUDA or MPS)
            mask = mask.unsqueeze(1).to(device) # move the mask to the device. Unsqueeze to add a dimension to the mask tensor (1 channel) since the model expects 1 channel masks (white and black where white is the object)  
        
            with torch.amp.autocast(device_type=device): # use torch.amp.autocast to use mixed precision training which means it uses both 16-bit and 32-bit floating point numbers to speed up training
                prediction = model(image) # get the prediction from the model using the image
                loss = loss_fn(prediction, mask) # calculate the loss using the prediction and the mask
                total_loss += loss.item() # add the loss to the total loss
    
    model.train() # set the model back to training mode
    
    return total_loss/len(data_loader) # return the average loss over the validation set
    

def save_checkpoint(model, optimizer, filename):
    '''
    This function saves the model and optimizer state dictionaries to a file.
    
    model: the model we are saving
    optimizer: the optimizer we are saving
    filename: the name of the file we are saving the model and optimizer to

    It's important to note that when you save a model and optimizer, the device the model is on is saved as well. So when you load the model and optimizer, you should load them onto the same device to avoid errors.
    '''
    model.eval() # set the model to evaluation mode so that it doesn't update weights
    checkpoint = { # create a dictionary containing the model and optimizer state dictionaries
        "state dict": model.state_dict(), # model state dictionary which is the model's parameters
        "optimizer": optimizer.state_dict(), # optimizer state dictionary which is the optimizer's parameters
    }
    torch.save(checkpoint, filename) # save the checkpoint to the filename in the current directory
    model.train() # set the model back to training mode

def retrieve_checkpoint(filename, model, optimizer, device):
    '''
    This function retrieves the model and optimizer state dictionaries from a file and loads them into the model and optimizer.

    filename: the name of the file we are retrieving the model and optimizer from
    model: the model we are loading the state dictionary into
    optimizer: the optimizer we are loading the state dictionary into
    device: the device we are using (CPU (worst case) or CUDA (NVIDIA) or MPS (macOS))
    '''
    checkpoint = torch.load(filename, map_location = device) # load the checkpoint from the filename into the device. As mentioned above, the device the model is on is saved as well so we need to load it onto the same device.
    model.load_state_dict(checkpoint["state dict"]) # load the model state dictionary into the model
    optimizer.load_state_dict(checkpoint["optimizer"]) # load the optimizer state dictionary into the optimizer
    model.train() # set the model back to training mode in case it was in evaluation mode

def example_images(loader, model, folder, device, epoch_num):
    '''
    This function saves example images, masks, and predictions to a folder. It's useful for visualizing the model's performance.

    loader: the data loader we are using (validation data loader)
    model: the model we are using (UNet in this case)
    folder: the folder we are saving the images to
    device: the device we are using (CPU (worst case) or CUDA (NVIDIA) or MPS (macOS))
    epoch_num: the epoch number we are saving the images for (useful for tracking progress)
    '''
    model.eval() # set the model to evaluation mode so that it doesn't update weights
    for index, (image, mask) in enumerate(loader): # iterate over the data loader (validation set) which contains images and masks in batches
        image = image.to(device) # move the image to the device (CPU or CUDA or MPS)
        mask = mask.unsqueeze(1).to(device) # move the mask to the device. Unsqueeze to add a dimension to the mask tensor (1 channel) since the model expects 1 channel masks (white and black where white is the object)

        with torch.no_grad(): # use torch.no_grad() to disable gradient tracking since we are not updating weights and don't need to track gradients which takes memory
            prediction = torch.sigmoid(model(image)) # get the prediction from the model using the image and then apply the sigmoid function to get values between 0 and 1
            prediction = (prediction > 0.5).float() # threshold the prediction to get binary values (0 or 1) using 0.5 as the threshold. Anything above 0.5 is 1 (white) and anything below is 0 (black)
        
        if not os.path.exists(folder): # if the folder doesn't exist, create it
            os.makedirs(folder) # create the folder

        save_image(prediction, f"{folder}/prediction_{index}_epoch_{epoch_num}.png") # save the prediction to the folder with the index and epoch number
        save_image(mask, f"{folder}/mask_{index}_epoch_{epoch_num}.png") # save the mask to the folder with the index and epoch number
        save_image(image, f"{folder}/image_{index}_epoch_{epoch_num}.png") # save the image to the folder with the index and epoch number
    
    model.train() # set the model back to training mode

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def randomly_select_vals(num_vals, all_images_dir, all_masks_dir, train_images_dir, val_images_dir, train_masks_dir, val_masks_dir):
    clear_directory(train_images_dir) # clear the training images directory
    clear_directory(val_images_dir) # clear the validation images directory
    clear_directory(train_masks_dir) # clear the training masks directory
    clear_directory(val_masks_dir) # clear the validation masks directory

    all_images = os.listdir(all_images_dir) # get all the images in the directory and store them in a list
    selected_images = random.sample(all_images, num_vals) # randomly select num_vals images from the list of all images
    for image in all_images:
        renamed_mask = image.replace(".jpg", "_mask.png") # rename the mask to match the image name
        image_path = os.path.join(all_images_dir, image)
        mask_path = os.path.join(all_masks_dir, renamed_mask) # get the corresponding mask path for the image
        if image in selected_images:
            shutil.copy(image_path, os.path.join(val_images_dir, image))
            shutil.copy(mask_path, os.path.join(val_masks_dir, renamed_mask))
        else:
            shutil.copy(image_path, os.path.join(train_images_dir, image))
            shutil.copy(mask_path, os.path.join(train_masks_dir, renamed_mask))


def test_model(model_loc, image_loc, image_height, image_width, device):
    model = torch.load(model_loc, map_location=device, weights_only=False)  # Load the model from the specified location
    model = model.to(device)
    model.eval()
    image = Image.open(image_loc).convert("RGB")
    image = np.array(image)

    transform = A.Compose([
        A.Resize(height = image_height, width = image_width),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    transformed = transform(image=image)["image"]
    transformed = transformed.unsqueeze(0).to(device)
    
    with torch.no_grad():
        raw_pred = torch.sigmoid(model(transformed))
        bin_mask = (raw_pred > 0.5).float()

    save_image(bin_mask, "segmented.png")
    mask_np = bin_mask.squeeze().cpu().numpy().astype(np.uint8) * 255  # Convert to numpy array and scale to 0-255
    
    cleaned_mask = remove_noise_connected_components(mask_np, min_area=100)  # Remove noise from the mask
    cv2.imwrite("segmented_noiseremoved.png", cleaned_mask)  # Save the cleaned mask as an image
