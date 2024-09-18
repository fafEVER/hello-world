import torch
import torchvision.transforms
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

dataset = torchvision.datasets.CIFAR10("./dataset",train=False, download=True,
                                     transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64)

input = torch.tensor(([1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,3,1],
                     [5,2,3,1,1],
                     [2,1,0,1,1]))

input = torch.reshape(input,(-1,1,5,5))  # -1： 系统会自动计算出来BatchSize
print(input.shape)

class Ack(nn.Module):
    def __init__(self):
        super(Ack,self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self,input):
        output = self.maxpool1(input)
        return output

ack = Ack()

writer = SummaryWriter("logs_maxpool")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = ack(imgs)
    writer.add_images("output",output,step)
    step += 1

writer.close()

