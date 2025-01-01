#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: Diff-expert
@Name: map_encoder.py
"""
import torch
import torch.nn as nn
import pdb
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

transform = transforms.Compose([
        transforms.ToTensor()
    ])
image_size = (90, 165)
image_path = "MAP.png"
image = Image.open(image_path).convert("L") 
resize_transform = transforms.Resize(image_size)
image = resize_transform(image)
image.save("MAP_gray.png")
image_tensor = transform(image)#.to(device)




class Map_Encoder(nn.Module):
    def __init__(self, in_channel, channel = 128):
 
        super(CNNEncoder, self).__init__()   
        self.intial_channel = 32
        self.features = nn.Sequential(
            nn.Conv2d(in_channel,self.intial_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.intial_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(self.intial_channel , 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        add_ons_channels = 64
        self.epsilon = 1e-4
        self.add_ons = nn.Sequential(
            nn.Conv2d(add_ons_channels, channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )
        self.p_map = nn.Sequential(
                nn.Conv2d(add_ons_channels, channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel,  self.intial_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(self.intial_channel),
                nn.Sigmoid()
            )
        
        flattened_dim = self.intial_channel*channel
        self.fc1 = nn.Linear(flattened_dim, 128)
        
        
    def scale(self, x, dim):
        x = x - x.amin(dim, keepdim=True)
        return x / x.amax(dim, keepdim=True).clamp_min(self.epsilon)

    def sigmoid(self, x, omega=10, sigma=0.5):
        return torch.sigmoid(omega * (x - sigma))
        
    def get_p_map(self, x):
        p_map = F.relu(self.p_map[:-1](x))
        p_map = self.scale(p_map, tuple(range(1, p_map.ndim)))
        return self.sigmoid(p_map)
    
    def attn_map(self, x, p_map):
        p_size = x.flatten(2).shape[2]
        p_x = torch.einsum('bphw,bchw->bpc', p_map, x) / p_size
        return p_x
        
    def forward(self, x):
        x = self.features(x)
        f_x = self.add_ons(x)
        p_map = self.get_p_map(x)
        p_x = self.attn_map(f_x, p_map)
        x = p_x.view(p_x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x, p_map
        



