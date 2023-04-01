import torch
import os

DATASET_TRAIN_PATH = os.path.join("dataset", "train")
DATASET_TEST_PATH = os.path.join("dataset", "test")

IMAGE_DATASET_PATH = os.path.join(DATASET_TRAIN_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_TRAIN_PATH, "masks")

IMAGE_TEST_DATASET_PATH = os.path.join(DATASET_TEST_PATH, "images")
MASK_TEST_DATASET_PATH = os.path.join(DATASET_TEST_PATH, "masks")

EVAL_SPLIT = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIN_MEMORY = True if DEVICE == "cuda" else False

NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

INIT_LR = 0.0005
NUM_EPOCHS = 100
BATCH_SIZE = 1

INPUT_IMAGE_WIDTH = 512
INPUT_IMAGE_HEIGHT = 512
SCALE_FACTOR = 0.25

THRESHOLD = 0.4

BASE_OUTPUT = "output"

MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])