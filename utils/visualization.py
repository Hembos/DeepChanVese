import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import matplotlib
from random import random


def plot_image_with_mask(image_path: str, boxes: list, masks: np.ndarray) -> None:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0

    figure, ax = plt.subplots()

    ax.imshow(image)


    for i in range(len(boxes)):
        colors = [(random(),random(),random()) for i in range(2)]
        cmaps = matplotlib.colors.LinearSegmentedColormap.from_list('cmaps', colors, N=2)
        rect = patches.Rectangle((boxes[i][0], boxes[i][1]), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        masked = np.ma.masked_where(masks[i] == 0, masks[i])

        ax.imshow(masked, cmap=cmaps, interpolation='none', alpha=0.6)

    plt.show()


# def plot_image(origImage, predMask):
#     figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
#     ax[0].imshow(origImage)
#     # ax[1].imshow(labeled_mask)
#     ax[2].imshow(masks[0])
#     # ax[2].imshow(predMask)
#     ax[0].set_title("Image")
#     # ax[1].set_title("Original Mask")
#     ax[2].set_title("Predicted Mask")

#     for i in range(len(obj_ids)):
#         rect = patches.Rectangle((boxes[i][0], boxes[i][1]), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1], linewidth=1, edgecolor='r', facecolor='none')
#         ax[0].add_patch(rect)

#     tmp = origImage[boxes[0][1]:boxes[0][3],boxes[0][0]:boxes[0][2]]
#     ax[1].imshow(tmp)

#     figure.tight_layout()
#     plt.show()