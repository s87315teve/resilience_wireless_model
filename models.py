import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import dump, load
import time
import os
import math
import json 
import argparse
from torchvision import models
from torchvision.models import convnext_tiny
from torch_geometric.nn import GCNConv, global_mean_pool

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=4, num_encoder_layers=4, dim_feedforward=256, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, dim_feedforward)
        encoder_layers = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)
        self.output_linear = nn.Linear(dim_feedforward, output_dim)
        
    def forward(self, src):
        src = self.input_linear(src)
        src = src.unsqueeze(1)  
        output = self.transformer_encoder(src)
        output = output.squeeze(1)  
        output = self.output_linear(output)
        return output

class CustomResNet(nn.Module):
    def __init__(self, input_features, output_dim, resnet_version=18):
        super(CustomResNet, self).__init__()
        
        if resnet_version==18:
            self.resnet = models.resnet18(pretrained=False)
        elif resnet_version==50:
            self.resnet = models.resnet50(pretrained=False)

        
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0, bias=False)

        self.input_adapter = nn.Linear(input_features, 64)  # 将输入映射到 64 维

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, x):

        x = self.input_adapter(x)
        x = x.view(x.size(0), 1, 8, 8)  
        x = self.resnet(x)
        
        return x



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class FiLM(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super(FiLM, self).__init__()
        self.condition_encoder = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, feature, condition):
        condition = self.condition_encoder(condition)
        gamma, beta = torch.chunk(condition, 2, dim=1)
        return (1 + gamma) * feature + beta

class GeoSigNet(nn.Module):
    def __init__(self, input_dim=None, output_dim=1, dropout=0.1):
        super(GeoSigNet, self).__init__()
        
        self.feature_a = nn.Sequential(
            nn.Linear(6, 128),
            Swish(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            Swish(),
            nn.Dropout(dropout)
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.film = FiLM(256, 2)
        
        self.feature_b = nn.Sequential(
            nn.Linear(256, 512),
            Swish(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout)
        )
        
        self.feature_b_res = nn.Linear(256, 512)
        
        self.residual = nn.Sequential(
            nn.Linear(512 + 2, 512),
            Swish(),
            nn.BatchNorm1d(512)
        )
        
        self.final = nn.Sequential(
            nn.Linear(512, 256),
            Swish(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            Swish(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
        self.self_attention = nn.MultiheadAttention(512, 8, dropout=dropout)

    def forward(self, x):
        positions = x[:, :6]
        distances_angles = x[:, 6:8]
        freq_los = x[:, 8:]
        
        feature_a = self.feature_a(positions)
        feature_a = feature_a.unsqueeze(1)
        
        transformer_output = self.transformer_encoder(feature_a)
        transformer_output = transformer_output.squeeze(1)
        
        pooled_output = self.global_max_pool(transformer_output.unsqueeze(2)).squeeze(2)
        
        feature_b = self.film(pooled_output, distances_angles)
        feature_b = self.feature_b(feature_b)
        
        feature_b_res = self.feature_b_res(pooled_output)
        feature_b = feature_b + feature_b_res
        
        feature_b = feature_b.unsqueeze(0)
        attn_output, _ = self.self_attention(feature_b, feature_b, feature_b)
        feature_b = feature_b + attn_output
        feature_b = feature_b.squeeze(0)
        
        residual_input = torch.cat([feature_b, freq_los], dim=1)
        residual_output = self.residual(residual_input)
        final_input = feature_b + residual_output
        
        output = self.final(final_input)
        return output
#done
class GeoSigNet_Distance(nn.Module):
    def __init__(self, input_dim=None, output_dim=1, dropout=0.1):
        super(GeoSigNet_Distance, self).__init__()
        
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.film = FiLM(256, 2)
        self.feature_a = nn.Sequential(
            nn.Linear(1, 256),
            Swish(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout)
        )
        self.feature_b = nn.Sequential(
            nn.Linear(256, 512),
            Swish(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout)
        )
        
        self.feature_b_res = nn.Linear(256, 512)
        
        self.residual = nn.Sequential(
            nn.Linear(512 + 1, 512),
            Swish(),
            nn.BatchNorm1d(512)
        )
        
        self.final = nn.Sequential(
            nn.Linear(512, 256),
            Swish(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            Swish(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
        self.self_attention = nn.MultiheadAttention(512, 8, dropout=dropout)

    def forward(self, x):
        distances = x[:, 0:1]
        freq = x[:, 1:]
        
        feature_a = self.feature_a(distances)
        feature_b = self.feature_b(feature_a)
        
        feature_b_res = self.feature_b_res(feature_a)
        feature_b = feature_b + feature_b_res
        
        feature_b = feature_b.unsqueeze(0)
        attn_output, _ = self.self_attention(feature_b, feature_b, feature_b)
        feature_b = feature_b + attn_output
        feature_b = feature_b.squeeze(0)
        
        residual_input = torch.cat([feature_b, freq], dim=1)
        residual_output = self.residual(residual_input)
        final_input = feature_b + residual_output
        
        output = self.final(final_input)
        return output


#正在改
class GeoSigNet_DistanceAngle(nn.Module):
    def __init__(self, input_dim=None, output_dim=1, dropout=0.1):
        super(GeoSigNet_DistanceAngle, self).__init__()
        
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.film = FiLM(256, 2)
        self.feature_a = nn.Sequential(
            nn.Linear(2, 256),
            Swish(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout)
        )
        self.feature_b = nn.Sequential(
            nn.Linear(256, 512),
            Swish(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout)
        )
        
        self.feature_b_res = nn.Linear(256, 512)
        
        self.residual = nn.Sequential(
            nn.Linear(512 + 1, 512),
            Swish(),
            nn.BatchNorm1d(512)
        )
        
        self.final = nn.Sequential(
            nn.Linear(512, 256),
            Swish(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            Swish(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
        self.self_attention = nn.MultiheadAttention(512, 8, dropout=dropout)

    def forward(self, x):
        distances_angles = x[:, 0:2]
        freq = x[:, 2:]
        
        feature_a = self.feature_a(distances_angles)

        feature_b = self.feature_b(feature_a)
        
        feature_b_res = self.feature_b_res(feature_a)
        feature_b = feature_b + feature_b_res
        
        feature_b = feature_b.unsqueeze(0)
        attn_output, _ = self.self_attention(feature_b, feature_b, feature_b)
        feature_b = feature_b + attn_output
        feature_b = feature_b.squeeze(0)
        
        residual_input = torch.cat([feature_b, freq], dim=1)
        residual_output = self.residual(residual_input)
        final_input = feature_b + residual_output
        
        output = self.final(final_input)
        return output

class GeoSigNet_DistanceLOS(nn.Module):
    def __init__(self, input_dim=None, output_dim=1, dropout=0.1):
        super(GeoSigNet_DistanceLOS, self).__init__()
        
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.film = FiLM(256, 2)
        self.feature_a = nn.Sequential(
            nn.Linear(1, 256),
            Swish(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout)
        )
        self.feature_b = nn.Sequential(
            nn.Linear(256, 512),
            Swish(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout)
        )
        
        self.feature_b_res = nn.Linear(256, 512)
        
        self.residual = nn.Sequential(
            nn.Linear(512 + 2, 512),
            Swish(),
            nn.BatchNorm1d(512)
        )
        
        self.final = nn.Sequential(
            nn.Linear(512, 256),
            Swish(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            Swish(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
        self.self_attention = nn.MultiheadAttention(512, 8, dropout=dropout)

    def forward(self, x):
        distances = x[:, 0:1]
        freq_los = x[:, 1:]
        
        feature_a = self.feature_a(distances)
        feature_b = self.feature_b(feature_a)
        
        feature_b_res = self.feature_b_res(feature_a)
        feature_b = feature_b + feature_b_res
        
        feature_b = feature_b.unsqueeze(0)
        attn_output, _ = self.self_attention(feature_b, feature_b, feature_b)
        feature_b = feature_b + attn_output
        feature_b = feature_b.squeeze(0)
        
        residual_input = torch.cat([feature_b, freq_los], dim=1)
        residual_output = self.residual(residual_input)
        final_input = feature_b + residual_output
        
        output = self.final(final_input)
        return output

class GeoSigNet_DistanceGPS(nn.Module):
    def __init__(self, input_dim=None, output_dim=1, dropout=0.1):
        super(GeoSigNet_DistanceGPS, self).__init__()
        
        self.feature_a = nn.Sequential(
            nn.Linear(6, 128),
            Swish(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            Swish(),
            nn.Dropout(dropout)
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.film = FiLM(256, 1)
        
        self.feature_b = nn.Sequential(
            nn.Linear(256, 512),
            Swish(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout)
        )
        
        self.feature_b_res = nn.Linear(256, 512)
        
        self.residual = nn.Sequential(
            nn.Linear(512 + 1, 512),
            Swish(),
            nn.BatchNorm1d(512)
        )
        
        self.final = nn.Sequential(
            nn.Linear(512, 256),
            Swish(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            Swish(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
        self.self_attention = nn.MultiheadAttention(512, 8, dropout=dropout)

    def forward(self, x):
        positions = x[:, :6]
        distances = x[:, 6:7]
        freq = x[:, 7:]
        
        feature_a = self.feature_a(positions)
        feature_a = feature_a.unsqueeze(1)
        
        transformer_output = self.transformer_encoder(feature_a)
        transformer_output = transformer_output.squeeze(1)
        
        pooled_output = self.global_max_pool(transformer_output.unsqueeze(2)).squeeze(2)
        
        feature_b = self.film(pooled_output, distances)
        feature_b = self.feature_b(feature_b)
        
        feature_b_res = self.feature_b_res(pooled_output)
        feature_b = feature_b + feature_b_res
        
        feature_b = feature_b.unsqueeze(0)
        attn_output, _ = self.self_attention(feature_b, feature_b, feature_b)
        feature_b = feature_b + attn_output
        feature_b = feature_b.squeeze(0)
        
        residual_input = torch.cat([feature_b, freq], dim=1)
        residual_output = self.residual(residual_input)
        final_input = feature_b + residual_output
        
        output = self.final(final_input)
        return output


# def custom_loss(pred, target):
#     mse_loss = F.mse_loss(pred, target)
#     huber_loss = F.smooth_l1_loss(pred, target)
#     return 0.7 * mse_loss + 0.3 * huber_loss