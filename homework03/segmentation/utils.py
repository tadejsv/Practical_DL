from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .data import SegmentationDataset
from .trainer import setup_trainer
from .model import SegmentationModel

def train_val_set(transform_train=None, transform_val=None):
    """Returns the train and validation dataset, with optional transformations applied"""
    
    train_folder_img = 'BBBC018_v1_images-fixed/train'
    train_folder_mask = 'BBBC018_v1_outlines/train'
    
    val_folder_img = 'BBBC018_v1_images-fixed/val'
    val_folder_mask = 'BBBC018_v1_outlines/val'
    
    train_set = SegmentationDataset(train_folder_img, train_folder_mask, transform_train)
    val_set = SegmentationDataset(val_folder_img, val_folder_mask, transform_val)
    
    return train_set, val_set

def spatial_aug():
    """Returns the list of spatial augmentations"""
    
    augs = [
        A.VerticalFlip(),
        A.HorizontalFlip(),
        A.ElasticTransform(p=0.75),
        A.Transpose(),
        A.Rotate(limit=180, p=1),
        A.RandomSizedCrop(min_max_height=(360,400), height=384, width=384),
    ]
    
    return augs
    
def color_aug():
    """Returns the list of color augmentations"""
    
    augs = [
        A.GaussianBlur(),
        A.HueSaturationValue(),
    ]
    
    return augs
    
def processing_aug():
    """A list of Float and ToTensor augmentations"""
    
    mask_to_float = A.Lambda(mask=...)
    
    augs = [
        A.ToFloat(),
        mask_to_float,
        ToTensorV2()
    ]
    
    return augs
    
def prepare_data_loaders(batch_size):
    """Return the data loaders for train and validation sets. Train set had spatial and color augmentations applied."""
    
    # Get datasets with augmentations
    train_augs = A.Compose(spatial_aug() + color_aug() + processing_aug())
    val_augs = A.Compose(processing_aug())
    
    train_set, val_set = train_val_set(train_augs, val_augs)
    
    # Get dataloaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=8, pin_memory=True
    )
    
    return train_loader, val_loader