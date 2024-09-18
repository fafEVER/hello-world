import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)


dataloader = DataLoader(dataset,batch_size=64)


class Ack(nn.Module):
    def __init__(self):
        super(Ack,self).__init__()
        self.linear1 = Linear(196608,10)

    def forward(self,input):
        output = self.linear1(input)
        return output

ack = Ack()


for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)    #flatten:降为一行数据（摊平作为全连接层的输入）
    print(output.shape)
    output = ack(output)
    print(output)
