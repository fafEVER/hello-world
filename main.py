from torch.utils.data import Dataset
from PIL import Image
import  os

class MyData(Dataset):

    def __init__(self,root_dir,lable_dir):
        self.root_dir = root_dir
        self.lable_dir = lable_dir
        self.path = os.path.join(self.root_dir,self.lable_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.lable_dir,img_name)
        img = Image.open(img_item_path)
        lable = self.lable_dir
        return img,lable

    def __len__(self):
        return len(self.img_path)

root_dir = "data_1/train"
ants_lable_dir = "ants_image"
bees_lable_dir = "bees_image"
ants_dataset = MyData(root_dir,ants_lable_dir)
bees_dataset = MyData(root_dir,bees_lable_dir)

train_dataset = ants_dataset + bees_dataset
import os

root_dir = 'data_1\\train'
target_dir = 'ants_image'
img_path = os.listdir(os.path.join(root_dir, target_dir))
lable = target_dir.split('_')[0]
out_dir = 'ants_lable'
for i in img_path:
    file_name = i.split('.jpg')[0]
    with open(os.path.join(root_dir, out_dir,"{}.txt".format(file_name)),'w') as f:
        f.write(lable)