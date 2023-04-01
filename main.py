from unet import train, make_predictions
from config import *
import numpy as np
from utils import plot_image_with_mask
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image


is_train = False
prec = []

def measure_precision(predicted_masks, gt_mask_path):
    gt_mask = np.asarray(Image.open(gt_mask_path))
    
    if gt_mask.max() == 255 and predicted_masks.shape[0] != 0:
        pred_mask = predicted_masks.max(axis=0)

        prec.append(np.sum(pred_mask == gt_mask) / (pred_mask.shape[0] * pred_mask.shape[1]))
    else:
        prec.append(1.0)

if __name__ == "__main__":
    if is_train:
        train()
    else:
        images_paths = list(sorted(os.listdir(IMAGE_DATASET_PATH)))
        masks_paths = list(sorted(os.listdir(MASK_DATASET_PATH)))

        for image_name, mask_name in tqdm(zip(images_paths, masks_paths), total=len(images_paths)):
            image_path =  os.path.join(IMAGE_DATASET_PATH, image_name)
            mask_path = os.path.join(MASK_DATASET_PATH, mask_name)
            boxes, masks = make_predictions(image_path)
            measure_precision(masks, mask_path)
            plot_image_with_mask(image_path, boxes, masks)
        
        plt.plot(prec, range(len(images_paths)))
        plt.show()