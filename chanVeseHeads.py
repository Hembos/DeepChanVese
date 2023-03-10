import torch.nn as nn
import skfmm

from chanvese import ChanVese

class InitialTSDFModel(nn.Module):
    def __init__(self) -> None:
        super(InitialTSDFModel, self).__init__()

    def forward(self, input):
        sd = skfmm.distance(input)
        sd_max = sd.max()
        sd_min = sd.min()
        sd = -1 + (sd - sd_min) / (sd_max - sd_min) * 2

        sd = nn.Upsample(scale_factor=4, mode='bilinear')(sd)

        return sd

class HyperParameterModel(nn.Module):
    def __init__(self, iterations_num, image_size):
        super(HyperParameterModel, self).__init__()

        self.num_params = iterations_num * 2 + 3

        self.conv1 = nn.Conv2D(1, 32, 3, 1)
        self.avg1 = nn.AvgPool2d(3, 2)

        self.conv2 = nn.Conv2D(32, 64, 3, 1)
        self.avg2 = nn.AvgPool2d(3, 2)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(image_size * image_size * 64, 2 * self.num_params)
        self.fc2 = nn.Linear(2 * self.num_params, self.num_params)

        self.sigmoidLayer = nn.Sigmoid()

    def forward(self, input):
        x = self.conv1(input)
        x = self.avg1(x)

        x = self.conv2(x)
        x = self.avg2(x)

        x = self.flat(x)

        x = self.fc1(x)
        x = self.fc2(x)

        x = self.sigmoidLayer(x)
        x *= 2

        return x

class ChanVeseFeaturesModel(nn.Module):
    def __init__(self):
        super(ChanVeseFeaturesModel, self).__init__()

        self.conv1 = nn.Conv2D(1, 15, 3, 1)

        self.conv2 = nn.Conv2D(15, 30, 3, 1)

        self.conv3 = nn.Conv2D(30, 45, 4, 1)

        self.conv4 = nn.Conv2D(45, 64, 5, 1)

        self.ups1 = nn.Upsample(scale_factor=3, mode='bilinear')
        self.ups2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, input):
        x = self.conv1(input)

        x = self.ups1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.ups2(x)

        return x
    
class ChanVeseModel(nn.Module):
    def __init__(self, num_iters):
        super(ChanVeseModel, self).__init__()

        self.initialTSDFModel = InitialTSDFModel()
        self.hyperParameterModel = HyperParameterModel(num_iters)
        self.chanVeseFeaturesModel = ChanVeseFeaturesModel()

        self.chanVese = ChanVese(num_iters)

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

        return self.chanVese.run(features, initTSDF, dt, eps, lam1, lam2, mu)