# Based on https://github.com/FarrandTom/deep-learning-fairness
# from @article{farrand2020neither, title={Neither Private Nor Fair: Impact of Data Imbalance on Utility and Fairness in Differential Privacy}, author={Farrand, Tom and Mireshghallah, Fatemehsadat and Singh, Sahib and Trask, Andrew}, journal={arXiv preprint arXiv:2009.06389}, year={2020} }


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def reseed(seed=5):
    """
    Reseed all relevant seeds.
    """
    seed = 5
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


class SimpleNet(nn.Module):
    def __init__(self, name=None, created_time=None):
        super(SimpleNet, self).__init__()
        self.created_time = created_time
        self.name=name
        reseed()


class Net(SimpleNet):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
