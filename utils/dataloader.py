import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import rasterio
import glob
import os
import matplotlib.pyplot as plt



# Dataset Definition

class BuildingDataset(Dataset):
    def __init__(self, image_mask_pair):
        self.image_mask_pair = image_mask_pair
        
    def __repr__(self):
        return f"Dataset(n_image_mask_pair = {len(self.image_mask_pair)})"

    def __len__(self):
        return len(self.image_mask_pair)
    
    
    def pad_image(self,image, x):
        padding = (x, x, x, x)
        padded_image = F.pad(image, padding, mode='constant', value=0)
        return padded_image

    
    def __getitem__(self, idx):
        image_path, mask_path = self.image_mask_pair[idx]

        # Read image
        with rasterio.open(image_path) as src:
            image = src.read()  # Read the image bands
            image = image[0:3,:,:]
            
        if image is None:
            raise ValueError(f"Image at {image_path} could not be read.")
        
        image = image / 255.0
        image = torch.from_numpy(image.astype(np.float32))

        # Read mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # Read the first band
        if mask is None:
            raise ValueError(f"Mask at {mask_path} could not be read.")
        mask = np.expand_dims(mask, 0) / 255.0
        mask = torch.from_numpy(mask.astype(np.float32))

        return self.pad_image(image,10), self.pad_image(mask,  10)
    
    

if __name__== '__main__':
        
    # Create image-mask pairs
    image_folder_path = './sample_datasets/image'
    mask_folder_path = './sample_datasets/mask'

    mask_files = {os.path.splitext(os.path.basename(path))[0]: path for path in glob.glob(os.path.join(mask_folder_path, '*.tif'))}

    print(f"Found {len(mask_files)} mask files.")
    if len(mask_files) == 0:
        print("No mask files found. Please check the mask folder path and file extensions.")

    image_mask_pair = [
        (os.path.join(image_folder_path, os.path.basename(image_path)),
        mask_files[os.path.splitext(os.path.basename(image_path))[0]])
        for image_path in glob.glob(os.path.join(image_folder_path, '*.tif'))
        if os.path.splitext(os.path.basename(image_path))[0] in mask_files
    ]

    print(f"Found {len(image_mask_pair)} image-mask pairs.")

    if len(image_mask_pair) == 0:
        print("No image-mask pairs found. Exiting.")
        

    # Dataset and DataLoader
    dataset = BuildingDataset(image_mask_pair=image_mask_pair)
    print(dataset)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    for batch in train_loader:
        print(batch)
    
    