import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from utils import load_data_2D, BATCH_SIZE, SEG_TRAIN_PATH, IMAGE_TRAIN_PATH

class ImageDataset(Dataset):
    """
    Stores images and their corresponding segmentation masks and applies transformations specified
    Overrides __len__ and __getitem__ to allow for use with torch.utils.DataLoader
    """
    def __init__(self, image_dir, segmask_dir, transforms, early_stop=False):
        image_names = sorted([os.path.join(image_dir, i) for i in os.listdir(image_dir)])
        mask_names = sorted([os.path.join(segmask_dir, m) for m in os.listdir(segmask_dir)])

        self.transforms = transforms

        # load nii files using utils helper
        # don't need to specify dimensions unless not using HIPmri
        self.images = load_data_2D(image_names, normImage=True, early_stop=early_stop)
        self.seg_masks = load_data_2D(mask_names, normImage=False, early_stop=early_stop)   

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # load images into torch tensors before returning
        img = self.images[idx]
        msk = self.seg_masks[idx]

        # convert to tensors
        img = torch.tensor(img)
        msk = torch.tensor(msk)

        # add channel dimension, conv in torch expects 4d input including
        # 1 channel as images are greyscale 
        img = img.unsqueeze(0)
        msk = msk.unsqueeze(0)

        if self.transforms:
            # using transforms vs
            img, msk = self.transforms(img, msk)

        return (img, msk)
    
def get_loader(image_dir, mask_dir, early_stop=False, transforms=None):
    """
    Takes directory of image slices and corresponding segmentation masks
    and returns a DataLoader, to be used for training and running inference on model
    """
    dataset = ImageDataset(image_dir, mask_dir, transforms=transforms, early_stop=early_stop)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return loader

if __name__ == "__main__":



    loader = get_loader(IMAGE_TRAIN_PATH, SEG_TRAIN_PATH, early_stop=False)

    print(len(loader))