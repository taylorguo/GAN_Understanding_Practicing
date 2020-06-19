import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # self.d = 128
        self.depths = [1024, 512, 256, 128, ]
        self.deconv1 = torch.nn.ConvTranspose2d()