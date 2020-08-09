import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .data import SegmentationDataset

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
    
    augs = [
        A.ToFloat(),
        ToTensorV2()
    ]
    
def prepare_data_loaders():
    """Return the data loaders for train and validation sets. Train set had spatial and color augmentations applied."""
    
    train_augs = A.Compose(spatial_aug() + color_aug() + processing_aug())
    val_augs = A.Compose(processing_aug())
    
    train_set, val_set = train_val_set(train_augs, val_augs)