from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from train_val_test import train_model
from train_val_test import test_model

cudnn.benchmark = True
plt.ion()   # interactive mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 预训练模型
model_conv = models.resnet18(pretrained=False)
model_conv.load_state_dict(torch.load('./pertrained_model/resnet18-f37072fd.pth'))

for param in model_conv.parameters():
    param.requires_grad = False

# 新模块参数模型默认为requires_grad=True
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

# 只复制最后一层conv
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# 只优化最后一层的参数
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# 每7个epoch进行一次衰减
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

test_model(model_conv, criterion)