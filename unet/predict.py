from config import *
import numpy as np
import torch
import cv2
import os
from utils import run_labeling
from chan_vese import exec_chan_vese
from typing import Tuple

from time import time


def apply_chan_vese(init_mask: np.ndarray, image: np.ndarray) -> Tuple[list, np.ndarray]:
    mask = init_mask == 255
    labeled_mask = run_labeling(mask.flatten(), mask.shape[0], mask.shape[1], 0)
    labeled_mask = np.resize(labeled_mask, (mask.shape[0], mask.shape[1]))

    obj_ids = np.unique(labeled_mask)
    obj_ids = obj_ids[1:]

    masks = labeled_mask == obj_ids[:, None, None]
    boxes = []
    for i in range(len(obj_ids)):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

        clipped_image = image[ymin:ymax,xmin:xmax][:,:,:1]
        phi = exec_chan_vese(clipped_image.astype(np.float64).flatten(), clipped_image.shape[1], clipped_image.shape[0], masks[i].astype(np.float64)[ymin:ymax,xmin:xmax].flatten(), 1000)
        phi = np.reshape(phi, (clipped_image.shape[0], clipped_image.shape[1]))

        masks[i][ymin:ymax,xmin:xmax] = phi >= 0

    return (boxes, masks)
	
def make_predictions(imagePath: str) -> Tuple[list, np.ndarray]:
    model = torch.load(MODEL_PATH).to(DEVICE)

    model.eval()
    with torch.no_grad():
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        orig = image.copy()
        image = cv2.resize(image, (round(INPUT_IMAGE_HEIGHT * SCALE_FACTOR), round(INPUT_IMAGE_WIDTH * SCALE_FACTOR)))

        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(DEVICE)

        predMask = model(image)
        predMask = torch.nn.Upsample(scale_factor=4, mode='nearest')(torch.sigmoid(predMask)).squeeze()
        predMask = predMask.cpu().numpy()

        predMask = (predMask > THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)

        boxes, final_masks = apply_chan_vese(predMask, orig)

        return (boxes, final_masks)