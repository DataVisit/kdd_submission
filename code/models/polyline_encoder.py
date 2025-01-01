#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: Diff-expert
@Name: polyline_encoder.py
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class PointEncoder(nn.Module):
    def __init__(self, config):
        super(PointEncoder, self).__init__()
        self.config = config
        in_dim = config.in_dim
        out_dim = config.out_dim
        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        if self.enable_temporal_encoding:
            x = x + self.temporal_encoding

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)

        x = x.view(-1, self.config.out_dim)
        return x

class MLPPointEncoder(nn.Module):
    def __init__(self, config):
        super(MLPPointEncoder, self).__init__()
        self.config = config
        in_dim = config.in_dim * 2
        out_dim = config.out_dim
        hidden_dim = config.hidden_dim
        self.n_layers = config.n_hidden_layers
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        hidden_layers = []
        for i in range(self.n_layers):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            hidden_layers.append(nn.ReLU())
        self.fc_hidden = nn.Sequential(*hidden_layers)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        B, D, T = x.shape
       
        x = x.reshape(B, D*T)
        x = F.relu(self.fc_in(x))
        if self.n_layers > 0:
            x = self.fc_hidden(x)
        x = self.fc_out(x)
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        in_dim = config.in_dim
        out_dim = config.out_dim
        hidden_dim = config.hidden_dim
        self.n_layers = config.n_hidden_layers
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        hidden_layers = []
        for i in range(self.n_layers):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            hidden_layers.append(nn.ReLU())
        self.fc_hidden = nn.Sequential(*hidden_layers)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        if self.n_layers > 0:
            x = self.fc_hidden(x)
        x = self.fc_out(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

class Polyline_Encoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super(Polyline_Encoder, self).__init__()
        self.config = config
       
        self.hidden_dim = config.model.d_model
        self.harbor_encoder = MLPPointEncoder(config.harbor_encoder)
        self.poly_encoder = PointEncoder(config.poly_encoder)
        encoder_layer = nn.TransformerEncoderLayer( \
            d_model=config.model.d_model, \
            nhead=config.model.nhead,  \
            dim_feedforward=config.model.dim_feedforward, \
            batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder( \
            encoder_layer, \
            num_layers=config.model.num_layers)
        
    def forward(self, ports, polylines):
        ports = ports  # (B, N, T, D)
        polylines = polylines  # (B, Nm, n, D)

        # Reshape the ports tensor and extract the port types
        B, N, T, D = ports.shape
        ports = ports.reshape(B * N, T, D)
        ports = ports.permute(0, 2, 1) 
      
        # Encode the ports
        ports = self.harbor_encoder(ports) 
        processed_ports = ports.reshape(B, N, -1)  # (B, N, D)

        polylines = torch.nan_to_num(polylines, nan=0)
        B, Nm, Np, D = polylines.shape 
        polylines = polylines.reshape(B*Nm, Np, D)
        polylines = polylines.permute(0, 2, 1)
        processed_polylines = self.poly_encoder(polylines) #(B, Nm, D)
        processed_polylines = processed_polylines.reshape(B, Nm, -1) #(B, Nm, D)

        obs_tokens = torch.cat([processed_ports, processed_polylines], dim=1)
        encoded_obs = self.transformer_encoder(obs_tokens)
    
        return encoded_obs
    
