from turtle import forward
import torch
import numpy as np
import torchvision
from torchvision import models
import torch.nn as nn

class vgg19_vis(nn.Module):

    def __init__(self, img_fc1_out=2742, img_fc2_out=32, dropout_p=0.4):
        super(vgg19_vis, self).__init__()

        vgg = models.vgg19(pretrained = False)
        vgg.load_state_dict(torch.load('./pertrained_model/vgg19-dcbb9e9d.pth'))

        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:1])
        
        self.vis_encoder = vgg

        self.vis_enc_fc1 = torch.nn.Linear(4096, img_fc1_out)

        self.vis_enc_fc2 = torch.nn.Linear(img_fc1_out, img_fc2_out)

        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, images):
        
        x = self.vis_encoder(images)

        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc1(x))
        )

        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc2(x))
        )

        x = torch.nn.functional.relu(x)
        
        prediction = torch.sigmoid(x)

        return prediction
