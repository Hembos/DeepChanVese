from dataset import SegmentationDataset
from unet import UNet
from config import *
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

def train() -> None:
    transforms = T.Compose([T.Resize((round(INPUT_IMAGE_HEIGHT * SCALE_FACTOR), round(INPUT_IMAGE_WIDTH * SCALE_FACTOR))),
    T.ToTensor()])

    transforms_mask = T.Compose([T.Resize((round(INPUT_IMAGE_HEIGHT * SCALE_FACTOR), round(INPUT_IMAGE_WIDTH * SCALE_FACTOR))),
    T.ToTensor()])

    dataset = SegmentationDataset(transforms, transforms_mask)

    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    num_eval_images = round(EVAL_SPLIT * len(dataset))
    dataset_train = torch.utils.data.Subset(dataset, indices[num_eval_images:])
    dataset_eval = torch.utils.data.Subset(dataset, indices[:num_eval_images])

    trainLoader = DataLoader(dataset_train, shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0)

    evalLoader = DataLoader(dataset_eval, shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=0)

    unet = UNet(outSize=(round(INPUT_IMAGE_HEIGHT * SCALE_FACTOR), round(INPUT_IMAGE_WIDTH * SCALE_FACTOR))).to(DEVICE)
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=INIT_LR)
    trainSteps = len(dataset_train) // BATCH_SIZE
    evalSteps = len(dataset_eval) // BATCH_SIZE

    H = {"train_loss": [], "eval_loss": []}

    startTime = time.time()
    for e in range(NUM_EPOCHS):
        print(f"epoch: {e}")

        unet.train()

        totalTrainLoss = 0
        totalEvalLoss = 0

        for (i, (x, y)) in tqdm(enumerate(trainLoader), total=trainSteps):
            (x, y) = (x.to(DEVICE), y.to(DEVICE))

            pred = unet(x)
            loss = lossFunc(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            totalTrainLoss += loss
            
        with torch.no_grad():
            unet.eval()
            for (x, y) in evalLoader:
                (x, y) = (x.to(DEVICE), y.to(DEVICE))
                pred = unet(x)
                totalEvalLoss += lossFunc(pred, y)

        avgTrainLoss = totalTrainLoss / trainSteps
        avgEvalLoss = totalEvalLoss / evalSteps

        print(f"train loss: {avgTrainLoss}")
        print(f"eval loss: {avgEvalLoss}")

        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["eval_loss"].append(avgEvalLoss.cpu().detach().numpy())

    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["eval_loss"], label="eval_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH)

    torch.save(unet, MODEL_PATH)
