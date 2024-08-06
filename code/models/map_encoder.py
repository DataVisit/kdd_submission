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
image_size = (180, 330)
image_path = "MAP.png"
image = Image.open(image_path).convert("L") 
resize_transform = transforms.Resize(image_size)
image = resize_transform(image)
image.save("MAP_gray.png")
image_tensor = transform(image)#.to(device)




class CNNEncoder(nn.Module):
    def __init__(self, in_channel, size, channel = 128):
 
        super(CNNEncoder, self).__init__()
        self.size = size
    
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
                nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(channel),
                nn.Sigmoid()
            )
        
        flattened_dim = channel*channel
        self.fc1 = nn.Linear(flattened_dim, 256)
        
        
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
        


class Map_Encoder(Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = (180, 330)
        self.x_range = config.map.lat_range
        self.y_range = config.map.lon_range
        self.encoder_dim = 128
        self.conv = CNNEncoder(in_channel=1, size= self.image_size)

    
    def generate(self, batch, image_tensor):
        image_map = image_tensor
        heat_map = self.generate_heatmap(batch, self.image_size, self.x_range, self.y_range)
        map_x_encoded = self.fuse_maps(image_map, heat_map)
        return map_x_encoded


    def generate_heatmap(self, x_t, image_size, x_range, y_range):
        x_scale = image_size[0] / (x_range[1] - x_range[0])
        y_scale = image_size[1] / (y_range[1] - y_range[0])

        x_pixel = image_size[0] - ((x_t[:, :, 0] - x_range[0]) * x_scale).long()
        y_pixel = ((x_t[:, :, 1] - y_range[0]) * y_scale).long()

        valid_indices = (~torch.isnan(x_t[:, :, 0])) & (~torch.isnan(x_t[:, :, 1]))
        heatmap = torch.zeros(x_t.shape[0], *image_size)

        size = 36
        sigma = 3.0
        kernel = torch.exp(-torch.arange(-size // 2, size // 2 + 1).float() ** 2 / (2 * sigma ** 2))
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()


        for i in range(x_t.shape[1]):
            for b in range(x_t.shape[0]):
                if not valid_indices[b, i]:
                    continue

                x_start = torch.clamp(x_pixel[b, i] - size // 2, 0, image_size[0] - size)
                y_start = torch.clamp(y_pixel[b, i] - size // 2, 0, image_size[1] - size)
                x_end = torch.clamp(x_pixel[b, i] + size // 2, 0, image_size[0])
                y_end = torch.clamp(y_pixel[b, i] + size // 2, 0, image_size[1])

                kernel_size_x = x_end - x_start
                kernel_size_y = y_end - y_start
                adjusted_kernel = kernel[:kernel_size_x, :kernel_size_y]
                heatmap[b, x_start:x_end, y_start:y_end] += adjusted_kernel

        max1, _ = heatmap.max(dim=1, keepdim=True)
        max2, _ = max1.max(dim=2, keepdim=True)
        heatmap = heatmap / (max2 + 1e-10)
        return heatmap

    def fuse_maps(self, image_map, heat_map, alpha=0.7):
        fused_map = alpha * heat_map + (1 - alpha) * image_map
        input = fused_map.unsqueeze(1).to(device)
        map_x_encoded, p_x = self.conv(input)
        return map_x_encoded
