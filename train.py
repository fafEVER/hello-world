import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *


#  准备数据集
train_data = torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                         download=True)
#  length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练集的长度为:{train_data_size}")
print(f"测试集的长度为:{test_data_size}")

#  利用DataLoader加载数据集
train_dataloader = DataLoader(train_data,64)
test_dataloader = DataLoader(test_data,64)

#  创建网络模型
ack =Ack_diff()

#  创建损失函数
loss_func = nn.CrossEntropyLoss()

#  优化器
learning_rate = 0.01
optim = torch.optim.SGD(ack.parameters(),lr=learning_rate)

#  设置训练网络的一些参数

total_train_step = 0  #记录训练次数
total_test_step = 0 #记录测试次数
epoch = 10 #训练轮数

writter = SummaryWriter("./logs_train")

for i in range(epoch):
    print(f"-------第{i+1}轮训练开始------")
    ack.train()
    for data in train_dataloader:
        imgs,targets = data
        outputs = ack(imgs)
        loss = loss_func(outputs,targets)
        # 优化器优化模型
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step += 1
        if total_train_step % 100 ==0:
            print(f"训练次数：{total_train_step}，Loss{loss.item()}")
            writter.add_scalar("train_loss",loss.item(),total_train_step)

    # 测试步骤开始
    ack.eval()
    total_test_loss = 0
    total_accuarcy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            outputs = ack(imgs)
            loss = loss_func(outputs,targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuarcy += accuracy

    print(f"整体测试集上的Loss:{total_test_loss}")
    print(f"整体测试集上的正确率:{total_accuarcy/test_data_size}")
    writter.add_scalar("test_loss",total_test_loss,total_test_step)
    writter.add_scalar("test_accuracy",total_accuarcy/test_data_size,total_test_step)
    total_test_step += 1


    torch.save(ack,f"ack_{i}.pth")
    print("模型已保存")

writter.close()




