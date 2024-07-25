# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, ChebConv

class Fusion_model(nn.Module):
    def __init__(self, pet_input_size, mri_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels):
        super(Fusion_model, self).__init__()
        self.low_branch = nn.Sequential(
            nn.BatchNorm1d(low_dim_input_size),
            nn.Linear(low_dim_input_size, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.mri_branch = nn.Sequential(
            nn.BatchNorm1d(mri_input_size),
            nn.Linear(mri_input_size, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.pet_branch = nn.Sequential(
            nn.BatchNorm1d(pet_input_size),
            nn.Linear(pet_input_size, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, output_dim)
        )

    def forward(self, pet_features, mri_features, low_dim_features,types):

        mri_features = self.mri_branch(mri_features)
        pet_features = self.pet_branch(pet_features)
        low_features = self.low_branch(low_dim_features)
        combined_features = torch.cat((mri_features, pet_features,low_features), dim=1)
        x = F.dropout(combined_features, p = 0.3)
        x = self.fusion_layer(combined_features)
        x = F.log_softmax(x, dim=1)
        return x


class Baseline_MLP1(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels, num_heads=8):
        super(Baseline_MLP1, self).__init__()
        self.bn_low = nn.BatchNorm1d(low_dim_input_size)
        self.linear_low = MLP(low_dim_input_size, embedding_dim)
        self.bn_high = nn.BatchNorm1d(high_dim_input_size)
        self.linear_high = MLP(high_dim_input_size, embedding_dim)
        self.bn_feature = nn.BatchNorm1d(embedding_dim*2)
        self.linear_feature = MLP(embedding_dim*2,embedding_dim)
        self.classfier = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, high_dim_features, low_dim_features):
        low_dim_embedded = self.linear_low(self.bn_low(low_dim_features))
        high_dim_features = self.linear_high(self.bn_high(high_dim_features))
        combined_features = torch.cat([high_dim_features, low_dim_embedded], dim=-1)
        x = combined_features
        x = self.linear_feature(self.bn_feature(x))
        y = self.classfier(x)
        y = F.softmax(y, dim=1)
        y = torch.log(y)
        return [y,x]
    
class Baseline_PET1(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels, num_heads=8):
        super(Baseline_PET1, self).__init__()
        self.bn_low = nn.BatchNorm1d(low_dim_input_size)
        self.linear_low = MLP(low_dim_input_size, embedding_dim)
        self.bn_high = nn.BatchNorm1d(high_dim_input_size)
        self.linear_high = MLP(high_dim_input_size, embedding_dim)
        self.bn_feature = nn.BatchNorm1d(embedding_dim*2)
        self.linear_feature = MLP(embedding_dim*2,embedding_dim)
        self.classfier = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, high_dim_features, low_dim_features):
        low_dim_embedded = self.linear_low(self.bn_low(low_dim_features))
        high_dim_features = self.linear_high(self.bn_high(high_dim_features))
        combined_features = torch.cat([high_dim_features, low_dim_embedded], dim=-1)
        x = combined_features
        x = self.linear_feature(self.bn_feature(x))
        y = self.classfier(x)
        y = F.softmax(y, dim=1)
        y = torch.log(y)
        return [y,x]
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim,output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        return self.bn(F.relu(self.linear(x)))

class Fusion_model(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels, num_heads=8):
        super(Fusion_model, self).__init__()
        self.bn_low = nn.BatchNorm1d(low_dim_input_size)
        self.linear_low = MLP(low_dim_input_size, embedding_dim)
        self.bn_high = nn.BatchNorm1d(high_dim_input_size)
        self.linear_high = MLP(high_dim_input_size, embedding_dim)
        self.bn_feature = nn.BatchNorm1d(embedding_dim*2)
        self.linear_feature = MLP(embedding_dim*2, embedding_dim)
        self.classfier = nn.Linear(embedding_dim + 2 * embedding_dim, output_dim)  # Updated size to account for mri_outputs and pet_outputs

        # Load pre-trained models for MRI and PET
        self.mri_model =  Baseline_MLP1(high_dim_input_size=498,
                     low_dim_input_size=17,
                     embedding_dim=128,
                     output_dim=2,
                     hidden_channels=128)
        self.mri_model.load_state_dict(torch.load('mri.pth'))
        self.pet_model = Baseline_MLP1(high_dim_input_size=166,
                     low_dim_input_size=17,
                     embedding_dim=128,
                     output_dim=2,
                     hidden_channels=128)
        self.pet_model.load_state_dict(torch.load('pet.pth'))

    def forward(self, high_dim_features, low_dim_features):
        mri_feature = high_dim_features[:, :498]
        pet_feature = high_dim_features[:, 498:]
        mri_outputs = self.mri_model(mri_feature,low_dim_features)
        pet_outputs = self.pet_model(pet_feature,low_dim_features)
        low_dim_embedded = self.linear_low(self.bn_low(low_dim_features))
        high_dim_features = self.linear_high(self.bn_high(high_dim_features))
        combined_features = torch.cat([high_dim_features, low_dim_embedded], dim=-1)
        all_outputs = self.linear_feature(self.bn_feature(combined_features))
        x = torch.cat([mri_outputs[1], pet_outputs[1], all_outputs], dim=-1)
        x = self.classfier(x)
        x = F.softmax(x, dim=1)
        x = torch.log(x)
        return x
