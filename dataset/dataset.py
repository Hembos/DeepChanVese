from torch.utils.data import Dataset
import os
from torchvision.transforms import transforms as T

from config import *

from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, transforms: T.Compose = None, transforms_mask: T.Compose = None) -> None:
        super().__init__()
        self.imagesPath = list(sorted(os.listdir(IMAGE_DATASET_PATH)))
        self.masksPath = list(sorted(os.listdir(MASK_DATASET_PATH)))
        self.transforms = transforms
        self.transforms_mask = transforms_mask

    def __len__(self):
        return len(self.imagesPath)
    
    def __getitem__(self, index: int):
        image_path = os.path.join(IMAGE_DATASET_PATH, self.imagesPath[index])
        mask_path = os.path.join(MASK_DATASET_PATH, self.masksPath[index])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms_mask(mask)

        return (image, mask)