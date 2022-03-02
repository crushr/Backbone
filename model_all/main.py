from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from transformers import AdamW, get_linear_schedule_with_warmup

from model import *
from dataloader import *
from train_val_test import *

cudnn.benchmark = True
plt.ion()   # interactive mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = 25
model = vgg19_vis().to(device)

criterion = nn.NLLoss()

# 优化器
optimizer = AdamW(model.parameters(),
                  lr = 3e-5,
                  eps = 1e-8)

total_steps = len(len(image_datasets['train'])) * EPOCHS

# lr衰减
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, 
                                            num_training_steps=total_steps)


model_conv = train_model(model, criterion, optimizer,
                         scheduler, num_epochs=25)
test_model(model_conv, criterion)







