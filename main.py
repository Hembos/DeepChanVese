from unet import train, make_predictions
from config import *
import numpy as np
from utils import plot_image_with_mask
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image
import json


data = {
    "tp": [],
    "fp": [],
    "tn": [],
    "fn": [],
    "accuracy": [],
    "bad": []
}

general_area = INPUT_IMAGE_HEIGHT * INPUT_IMAGE_WIDTH
images_paths = list(sorted(os.listdir(IMAGE_TEST_DATASET_PATH)))[100:]
masks_paths = list(sorted(os.listdir(MASK_TEST_DATASET_PATH)))[100:]

def measure_precision(predicted_masks, gt_mask_path):
    gt_mask = np.asarray(Image.open(gt_mask_path))
    
    accuracy = 1.0
    tp = 1.0
    tn = 0
    fp = 0
    fn = 0
    if gt_mask.max() == 255 and predicted_masks.shape[0] != 0:
        pred_mask = predicted_masks.max(axis=0)
        gt_mask = gt_mask == 255

        tp = np.sum(np.logical_and(pred_mask == 1, gt_mask == 1))
        tn = np.sum(np.logical_and(pred_mask == 0, gt_mask == 0))
        fp = np.sum(np.logical_and(pred_mask == 1, gt_mask == 0))
        fn = np.sum(np.logical_and(pred_mask == 0, gt_mask == 1))

        accuracy = np.sum(pred_mask == gt_mask) / (pred_mask.shape[0] * pred_mask.shape[1])

    data["accuracy"].append(accuracy)
    data["tp"].append(int(tp))
    data["tn"].append(int(tn))
    data["fp"].append(int(fp))
    data["fn"].append(int(fn))

    if accuracy < 0.85:
        data["bad"].append(gt_mask_path)

def calc_recall(tp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    return tp / (tp + fn)

def calc_precision(tp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return tp / (tp + fp)

def calc_f1(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    precision = precision[precision != 0.0]
    recall = recall[recall != 0.0]
    return 2 * precision * recall / (precision + recall)

def plot_precisions(data):
    plt.figure()
    plt.title("Accuracy")
    plt.boxplot(data["accuracy"])
    plt.savefig(PLOT_ACCURACY_PATH)

    plt.figure()
    plt.title("Recall")
    recall = calc_recall(np.array(data["tp"]), np.array(data["fn"]))
    plt.boxplot(recall)
    plt.savefig(PLOT_RECALL_PATH)

    plt.figure()
    plt.title("Precision")
    precision = calc_precision(np.array(data["tp"]), np.array(data["fp"]))
    plt.boxplot(precision)
    plt.savefig(PLOT_PRECISION_PATH)

    plt.figure()
    plt.title("F1")
    f1 = calc_f1(precision[:500], recall[:500])
    d = plt.boxplot(f1)
    # print(len(d["fliers"][0].get_data()[1]))
    # print(len(data["bad"]))
    plt.savefig(PLOT_F1_PATH)

    plt.show()

if __name__ == "__main__":
    if IS_TRAIN:
        train()
    else:
        for image_name, mask_name in tqdm(zip(images_paths, masks_paths), total=len(images_paths)):
            image_path =  os.path.join(IMAGE_TEST_DATASET_PATH, image_name)
            mask_path = os.path.join(MASK_TEST_DATASET_PATH, mask_name)
            boxes, masks = make_predictions(image_path)
            # measure_precision(masks, mask_path)
            plot_image_with_mask(image_path, boxes, masks)
        
        # with open(TEST_PRECISION_PATH, 'w') as f:
        #     json.dump(data, f)

        # with open(TEST_PRECISION_PATH) as f:
        #     data = json.load(f)

        # plot_precisions(data)