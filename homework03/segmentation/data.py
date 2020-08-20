import os

import skimage.io
import skimage

from torch.utils.data import Dataset

# Loader helper


def skimage_loader(path):
    img = skimage.img_as_ubyte(skimage.io.imread(path))
    return img


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')

# Another helper :)


def images_in_folder(folder):
    """Get a list of paths to images in the folder"""

    files = filter(lambda x: x.endswith(IMG_EXTENSIONS), os.listdir(folder))
    files = map(lambda x: os.path.join(folder, x), files)
    return list(files)


class SegmentationDataset(Dataset):
    """A data loader for segmentation data.

    It reads images and masks from a folder, and applies the (Albumenations) transformations to the pair.
    Assumes that images and masks have corresponding names (more specifically, the first word of the name).
    For example, this would be a valid combination of image/mask paths:

        images/
            object1-image.png
            object2-image.png
            ...

        maksk/
            object1-mask.jpg
            object2-mask.jpg
            ...


    Args:
        images_folder (string): The folder where images are
        masks_folder (string): The folder where masks are
        transform (callable): A transformation to be applied to image mask pairs.
    """

    def __init__(self, images_folder, masks_folder, transform=None):
        super().__init__()

        self.images = sorted(images_in_folder(images_folder))
        self.masks = sorted(images_in_folder(masks_folder))
        self.transform = transform

        assert len(self.images) == len(
            self.masks), "Length of images and masks should be the same"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, key):
        image = skimage_loader(self.images[key])
        mask = skimage_loader(self.masks[key])

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask
