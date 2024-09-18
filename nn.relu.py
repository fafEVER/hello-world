import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,-0.5],
                     [-1,3]])

input = torch.reshape(input,(-1,1,2,2))
print(input.shape)


dataset = torchvision.datasets.CIFAR10("./dataset",train=False,download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)


class Ack(nn.Module):
    def __init__(self):
        super(Ack, self).__init__()
        self.relu1 = ReLU(inplace=False)   #inpalce:是否替换原来的值(input),默认为False
        self.sigmoid1 = Sigmoid()

    def forward(self,input):
        output = self.sigmoid1(input)
        return output


ack = Ack()

writer = SummaryWriter("./logs_relu")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = ack(imgs)
    writer.add_images("output",output,step)
    step += 1

writer.close()





