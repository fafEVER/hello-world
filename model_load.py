import torch


# #方式一
# model = torch.load("vgg16_method1.pth")
# print(model)

#方式二
import torchvision
from torch import nn
from model_save import *
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
# print(vgg16)
# class AABBC(nn.Module):
#     def __init__(self):
#         super(AABBC,self).__init__()
#         self.conv1 = nn.Conv2d(3,64,3)
#
#     def forward(self,x):
#         x = self.conv1(x)
#         return x
model = torch.load("abc_method1.pth")
print(model)