import torch.nn as nn
import skfmm
import torch
import numpy as np

import torch.nn.functional as F

import matplotlib.pyplot as plt

import math

import chanvese_module

class InitialTSDFModel(nn.Module):
    def __init__(self) -> None:
        super(InitialTSDFModel, self).__init__()

    def forward(self, input):
        sd = skfmm.distance(input.cpu().detach().numpy())
        sd_max = sd.max()
        sd_min = sd.min()
        sd = -1 + (sd - sd_min) / (sd_max - sd_min) * 2

        sd = torch.unsqueeze(torch.from_numpy(sd), 0)
        sd = torch.unsqueeze(sd, 0)
        sd = nn.Upsample(scale_factor=4, mode='bilinear')(sd)

        return sd

class HyperParameterModel(nn.Module):
    def __init__(self, iterations_num, image_size):
        super(HyperParameterModel, self).__init__()

        self.num_params = iterations_num * 2 + 3

        self.conv1 = nn.Conv2d(1, 32, 2)
        image_size -= 1
        self.avg1 = nn.AvgPool2d(2, 2)
        image_size = math.floor((image_size - 2) / 2 + 1)
        self.conv2 = nn.Conv2d(32, 64, 2)
        image_size -= 1
        self.avg2 = nn.AvgPool2d(2, 2)
        image_size = math.floor((image_size - 2) / 2 + 1)
        self.fc1 = nn.Linear(image_size * image_size * 64, 2 * self.num_params)
        self.fc2 = nn.Linear(2 * self.num_params, self.num_params)

        self.sigmoidLayer = nn.Sigmoid()

    def forward(self, input):
        input = torch.unsqueeze(input, 0)
        x = self.avg1(F.relu(self.conv1(input)))

        x = self.avg2(F.relu(self.conv2(x)))

        x = torch.flatten(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.sigmoidLayer(x)
        x *= 2

        return x

class ChanVeseFeaturesModel(nn.Module):
    def __init__(self):
        super(ChanVeseFeaturesModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 2)

        self.conv2 = nn.Conv2d(16, 32, 2, padding=1)

        self.ups1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.ups2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input):
        input = torch.unsqueeze(input, 0)
        input = torch.unsqueeze(input, 0)
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))

        x = self.ups1(x)
        x = F.relu(self.conv3(x))

        x = self.ups2(x)

        return x
    
class ChanVeseModel(nn.Module):
    def __init__(self, num_iters):
        super(ChanVeseModel, self).__init__()

        self.initialTSDFModel = InitialTSDFModel()
        self.hyperParameterModel = HyperParameterModel(num_iters, 28)
        self.chanVeseFeaturesModel = ChanVeseFeaturesModel()

        # self.chanVese = ChanVese(num_iters)

        self.num_iters = num_iters

    def forward(self, input):
        initTSDF = self.initialTSDFModel(input)
        params = self.hyperParameterModel(input)
        features = self.chanVeseFeaturesModel(input)

        dt = params[:self.num_iters]
        eps = params[self.num_iters:2 * self.num_iters]
        lam1 = params[2*self.num_iters]
        lam2 = params[2*self.num_iters + 1]
        mu = params[-1]

        width = features.shape[2]
        height = features.shape[3]

        phi = chanvese_module.run(torch.squeeze(features).cpu().detach().numpy().astype(float).flatten(), width, height, 64, 
                            torch.squeeze(initTSDF).cpu().detach().numpy().astype(float).flatten(), self.num_iters, 
                            eps.cpu().detach().numpy().astype(float).flatten(), dt.cpu().detach().numpy().astype(float).flatten(), lam1.cpu().detach().numpy(), lam2.cpu().detach().numpy(), mu.cpu().detach().numpy())    

        # img_with_mask = np.zeros((height, width, 3))
        # for i in range(height):
        #     for j in range(width):
        #         img_with_mask[i][j] = (1, 1, 1) if phi[i * width + j] <= 0 else (1, 0, 0)

        # plt.imshow(img_with_mask)
        # plt.show()

        return np.resize(phi, (height, width))