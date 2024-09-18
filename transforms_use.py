from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法 -》 tensor数据类型
# 通过transforms.ToTensor去解决两个问题
img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1.transforms该如何使用
tensor_trans = transforms.ToTensor()  #创建自己的工具
tensor_img = tensor_trans(img)
# print(tensor_img)

writer.add_image("Tensor_img",tensor_img)

writer.close()
# 2.为什么需要Tensor数据类型