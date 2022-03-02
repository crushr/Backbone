from __future__ import print_function, division

import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os

cudnn.benchmark = True
plt.ion()   # interactive mode

# 数据扩充和规范化训练
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机长宽比裁剪
        transforms.RandomHorizontalFlip(),  # 依概率p水平翻转 无填充
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),      # 先resize再CenterCrop后可以得到一个不怎么会拉伸变形又包含大部分图像信息的正方形图片
        transforms.CenterCrop(224),  # 将图片从中心裁剪成224*224
        transforms.ToTensor(),       # 转换成为再张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化参数：各通道均值；各通道标准差
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '../ResNet18/weibo'

# datasets.ImageFolder利用构造图片数据加载器
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val','test']}
# 生成dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                            batch_size=5,
                                            shuffle=True,
                                            num_workers=2
                                            )
              for x in ['train', 'val','test']}
# dataset的大小
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}

class_names = image_datasets['train'].classes  # ImageFolder.classes储存了类别名称

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")