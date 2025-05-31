import torch # import PyTorch
import torch.nn as nn # import PyTorch's neural network module
from UNet import UNet # import the UNet class that we created in UNet.py
import albumentations as A # import Albumentations for image augmentation
from albumentations.pytorch import ToTensorV2 # import ToTensorV2 to convert images to tensors
from tqdm import tqdm # import tqdm for progress bars
from utils import * # import all functions from utils.py
import torch.optim as optim # import PyTorch's optimization module

'''Declaring Hyperparameters''' # Hyperparameters are variables that determine the network's architecture and how it learns
LEARNING_RATE = 1e-5 # Learning rate is the step size at which the model is updated. The smaller the learning rate, the slower the model learns but the more accurate it is.
NUM_WORKERS = 20 # Number of workers to use for loading data. Usually set to 2*number of cores.
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" # Device is the hardware on which the model is trained. Check if CUDA (NVIDIA GPU) is available, if not, check if MPS (macOS) is available, else use CPU.
BATCH_SIZE = 16 # Batch size is the number of images processed at once. 
NUM_EPOCHS = 100 # Number of epochs is the number of times the model sees the entire dataset. 
IMAGE_WIDTH = int(3024/6) # Image width is the width of the image. We divide by 6 to reduce the size of the image for faster training.
IMAGE_HEIGHT = int(4032/6) # Image height is the height of the image. We divide by 6 to reduce the size of the image for faster training.
PIN_MEMORY = True # Pin memory copies tensors to CUDA pinned memory for faster data transfer to GPU.
TRAIN_IMG_DIR = "data/training_images" # Directory containing training images.
TRAIN_MASK_DIR = "data/training_masks" # Directory containing training masks.
VAL_IMG_DIR = "data/val_images" # Directory containing validation images.
VAL_MASK_DIR = "data/val_masks" # Directory containing validation masks.
ALL_IMAGES_DIR = "data/all_images" # Directory containing all images.
ALL_MASKS_DIR = "data/all_masks" # Directory containing all masks.

print(f"Device is {DEVICE} \n") # print the device being used for our own information


'''Training UNet'''
def train(data_loader, model, optimizer, loss_fn, scaler):
    '''
    This function does one epoch of training on the training data loader. It calculates the loss and updates the model's weights.

    data_loader: training data loader (likely the one we created in create_loader)
    model: the model we are training (UNet in this case)
    optimizer: the optimizer we are using to update the model's weights (Adam in this case)
    loss_fn: the loss function we are using to calculate the loss (BCEWithLogitsLoss in this case)
    scaler: the gradient scaler we are using to scale the gradients (GradScaler in this case). 
    '''
    loading_bar = tqdm(data_loader) # tqdm is used to create a progress bar
    model.train() # set the model to training mode 
 
    for image, mask in loading_bar: # iterate over the training data loader
        image = image.to(device = DEVICE) # move the image to the device (GPU, MPS, or CPU)
        mask = mask.unsqueeze(1).to(device = DEVICE) # move the mask to the device. Unsqueeze to add a dimension to the mask tensor (1 channel) since the model expects 1 channel masks (white and black where white is the object)

        with torch.amp.autocast(device_type=DEVICE): # use torch.amp.autocast to use mixed precision training which means it uses both 16-bit and 32-bit floating point numbers to speed up training
            prediction = model(image) # get the prediction from the model using the image
            loss = loss_fn(prediction, mask) # calculate the loss using the loss function on the prediction and the mask

        optimizer.zero_grad() # zero the gradients to prevent accumulation

        # The next steps perform backpropagation. This is when the model learns from the loss by updating its weights. 
        # We use scaler because we are using mixed precision training which uses both 16-bit and 32-bit floating point numbers to speed up training. 
        # Scaler scales the loss to prevent underflow (which is when the number is too small to be represented by the computer) and overflow (which is when the number is too large to be represented by the computer).
        # These cases can arise when using 16-bit floating point numbers since they have a smaller range than 32-bit floating point numbers.

        scaler.scale(loss).backward() # This step stores the gradients of the loss in the model.parameters() attribute. The scale method scales the loss by a factor to prevent underflow and overflow.
        scaler.step(optimizer) # This step updates the model’s weights based on the gradients computed above using the optimizer. The gradients are unscaled before the optimizer step.
        scaler.update() # Updates the scaling factor for the next iteration 

        loading_bar.set_postfix(loss = loss.item()) # set the progress bar to display the loss for each batch

def main():
    '''
    This function is where we train the UNet model. We create the model, optimizer, loss function, and data loaders.
    '''

    # We use Albumentations for image augmentation. Image augmentation is a technique used to artificially increase the size of the training dataset by applying transformations to the images.

    # This is used to transform the images + their corresponding masks during training. 
    transform_image = A.Compose([ 
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), # Resize the image to the specified height and width
        A.Rotate(limit = 35, p = 0.3), # Rotate the image by a random angle between -35 and 35 degrees with a probability of 0.3
        A.Normalize( # Normalize the image which we need to do since the model expects pixel values between 0 and 1 to avoid exploding gradients
            mean=[0.0, 0.0, 0.0], # mean of the image for each channel (R, G, B)
            std=[1.0, 1.0, 1.0],  # standard deviation of the image for each channel (R, G, B)
            max_pixel_value=255.0 # maximum pixel value of the image
        ),
        ToTensorV2() # Convert the image to a PyTorch tensor
    ])

    # This is used to transform the validation images + their corresponding masks during validation.
    transform_val = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), # Resize the image to the specified height and width. It is important to resize the image to the same size as the training images.
        A.Normalize( 
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2() # Convert the image to a PyTorch tensor
    ])

    model = UNet(input_channels=3, output_channels=1).to(DEVICE) # Create the UNet model. The input channels are 3 (RGB) and the output channels are 1 (binary mask). We move the model to the device (GPU, MPS, or CPU).
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Create the optimizer. We use Adam optimizer. We pass the model parameters to the optimizer so that it can update the weights of the model. The learning rate is set to 1e-5.
    loss = nn.BCEWithLogitsLoss() # Create the loss function. We use Binary Cross Entropy with Logits Loss. We use with logits because the model outputs logits (raw scores) and we want to apply the sigmoid function to convert them to probabilities. 
    scaler = torch.amp.GradScaler(device=DEVICE) # Create the gradient scaler. We use GradScaler to scale the gradients to prevent underflow and overflow when using mixed precision training.

    randomly_select_vals(30, ALL_IMAGES_DIR, ALL_MASKS_DIR, TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR) 
    train_dataloader = create_train_dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, BATCH_SIZE, NUM_WORKERS, transform_image, pin_memory=PIN_MEMORY) 
    val_dataloader = create_val_dataset(VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, NUM_WORKERS, transform_val, pin_memory=PIN_MEMORY) # Create the validation data loader. We use the create_val_dataset function from utils.py to create the validation data loader.

    # Commented out since we are randomly selecting images for the val dataset

    # train_dataloader, val_dataloader = create_loader(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, NUM_WORKERS, transform_image, transform_val, pin_memory=PIN_MEMORY) # Create the data loaders. We use the create_loader function from utils.py to create the data loaders.

    for epoch in range(NUM_EPOCHS): # iterate over the number of epochs (times the model sees the entire dataset)
        model.train() # set the model to training mode
        train(train_dataloader, model, optimizer, loss, scaler) # train the model using the training data loader

        eval_loss = validate(val_dataloader, model, loss, DEVICE) # validate the model using the validate function from utils.py. This function calculates the loss on the validation set.
        print(f"Epoch: {epoch + 1}/{NUM_EPOCHS}, Validation Loss: {eval_loss}") # print the epoch number and the validation loss

        if epoch % 5 == 0: # save the model every 5 epochs
            save_checkpoint(model, optimizer, f"training_epoch_{epoch + 1}.pth.tar") # save the model and optimizer state dictionaries to a file. The file name is training_epoch_{epoch + 1}.pth.tar.
            example_images(val_dataloader, model, f"data/segmented_images/{epoch + 1}", DEVICE, (epoch + 1)) # save the segmented images from the validation data loader to a directory. The directory name is data/segmented_images/{epoch + 1}. 

        randomly_select_vals(30, ALL_IMAGES_DIR, ALL_MASKS_DIR, TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR) 
        train_dataloader = create_train_dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, BATCH_SIZE, NUM_WORKERS, transform_image, pin_memory=PIN_MEMORY)
        val_dataloader = create_val_dataset(VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, NUM_WORKERS, transform_val, pin_memory=PIN_MEMORY)
        
    torch.save(model, "final_tree_segmenting_UNET.pth") # save the final model to a file named final_tree_segmenting_UNET.pth. 

if __name__ == "__main__":
    main()