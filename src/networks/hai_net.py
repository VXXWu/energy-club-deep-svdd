import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class HAI_MLP(BaseNet):

    def __init__(self, n_features=79, window_size=64, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        self.input_dim = n_features * window_size
        
        self.fc1 = nn.Linear(self.input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256, eps=1e-04, affine=False)
        self.fc3 = nn.Linear(256, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.fc3(x)
        return x


class HAI_MLP_Autoencoder(BaseNet):

    def __init__(self, n_features=79, window_size=64, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        self.input_dim = n_features * window_size
        self.n_features = n_features
        self.window_size = window_size

        # Encoder (must match the Deep SVDD network above)
        self.fc1 = nn.Linear(self.input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256, eps=1e-04, affine=False)
        self.fc3 = nn.Linear(256, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.Linear(self.rep_dim, 256, bias=False)
        self.bn3 = nn.BatchNorm1d(256, eps=1e-04, affine=False)
        self.deconv2 = nn.Linear(256, 512, bias=False)
        self.bn4 = nn.BatchNorm1d(512, eps=1e-04, affine=False)
        self.deconv3 = nn.Linear(512, self.input_dim, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.fc3(x)
        
        x = self.deconv1(x)
        x = F.leaky_relu(self.bn3(x))
        x = self.deconv2(x)
        x = F.leaky_relu(self.bn4(x))
        x = self.deconv3(x)
        x = torch.sigmoid(x) # Assuming normalized data in [0, 1]

        return x.view(x.size(0), self.window_size, self.n_features)
