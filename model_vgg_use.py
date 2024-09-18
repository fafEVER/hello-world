import torchvision.datasets

# train_data = torchvision.datasets.ImageNet("./dataset_imagenet",split="train", download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor,
                                          download=True)

vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))
print(vgg16_true)

vgg16_false.classifier.add_module("7",nn.Linear(1000,50))
print(vgg16_false)
