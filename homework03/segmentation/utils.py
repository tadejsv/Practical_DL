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