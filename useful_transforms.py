from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")

img = Image.open("image/banaijian.png")
print(img)


# ToTensor的使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)
print(img_tensor)
print(img_tensor[0][0][0])
# Normalize  输入的是tensor数据类型
trans_norm = transforms.Normalize([0.3,0.3,0.3,0.3],[0.3,0.3,0.3,0.3])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])

writer.add_image("Normalize",img_norm,2)


# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)

writer.add_image("RESIZE",img_resize,0)


# Compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)

writer.add_image("RESIZE",img_resize_2,1)

# RandomCrop
trans_random = transforms.RandomCrop((200,100),padding_mode="edge")
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW",img_crop,i)


writer.close()