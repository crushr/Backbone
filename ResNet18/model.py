import torch
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
import torch.nn as nn
from transformers import BertModel

# 视觉
class VisionModel(nn.Module):
    def __init__(self, img_fc1_out=2742, img_fc2_out=32, dropout_p=0.4, fine_tune_module=False):
        super(VisionModel, self).__init__()
    
        vgg = models.vgg19(pretrained=True)
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:1])
        
        self.vis_model = vgg

        self.vis_enc_fc1 = torch.nn.Linear(4096, img_fc1_out)

        self.vis_enc_fc2 = torch.nn.Linear(img_fc1_out, img_fc2_out)

        self.fc = torch.nn.Linear(in_features=img_fc2_out, out_features=1)

        self.dropout = nn.Dropout(dropout_p)
        

    def forward(self, x):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        x = self.vis_model(x)

        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc1(x))
        )
        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc2(x))
        )
        prediction = torch.nn.functional.sigmoid(self.fc(x))

        return prediction

