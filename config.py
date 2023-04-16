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

THRESHOLD = 0.7

CHAN_VESE_ITER_NUM = 500
CHAN_VESE_USE_GPU = False

BASE_OUTPUT = "output"

MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt_old.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot_loss_cell.png"])
TEST_PRECISION_PATH = os.path.sep.join([BASE_OUTPUT, "test_precision_cell.json"])
PLOT_ACCURACY_PATH = os.path.sep.join([BASE_OUTPUT, "plot_accuracy_cell.png"])
PLOT_RECALL_PATH = os.path.sep.join([BASE_OUTPUT, "plot_recall_cell.png"])
PLOT_PRECISION_PATH = os.path.sep.join([BASE_OUTPUT, "plot_precision_cell.png"])
PLOT_F1_PATH = os.path.sep.join([BASE_OUTPUT, "plot_f1_cell.png"])

IS_TRAIN = False