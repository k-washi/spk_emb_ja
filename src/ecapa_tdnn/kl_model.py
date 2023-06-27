'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

References from https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py
'''

import math, torch
import torch.nn as nn

from src.ecapa_tdnn.model import SEModule, Bottle2neck


class ECAPA_TDNN_KL(nn.Module):

    def __init__(self, channel_size=1000, hidden_size=64, use_layer7=True):
        """
        Args:
            channel_size (int): channel size. Defaults to 1000.
            hidden_size (int): output hidden size. Defaults to 64.
        """

        super(ECAPA_TDNN_KL, self).__init__()
        self.conv1  = nn.Conv1d(80, channel_size, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(channel_size)
        self.layer1 = Bottle2neck(channel_size, channel_size, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(channel_size, channel_size, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(channel_size, channel_size, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*channel_size, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, hidden_size)
        self.fc6_mu = nn.Linear(3072, hidden_size)
        self.fc6_logvar = nn.Linear(3072, hidden_size)
        
        self.bn6 = nn.BatchNorm1d(hidden_size)
        
        self._use_layer7 = use_layer7
        if self._use_layer7:
            self.fc7 = nn.Linear(hidden_size, hidden_size)
            self.bn7 = nn.BatchNorm1d(hidden_size)

    def vectorize(self, x: torch.Tensor):
        mu, logvar = self.extract_feature(x)
        mu = self.sample(mu, logvar)
        mu = self.bn6(mu)
        return mu
    
    def sample(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            eps = torch.randn(size=mu.shape, device=mu.device)
            return mu + torch.exp(0.5 * logvar) * eps
        return mu
    
    def extract_feature(self, x: torch.Tensor) -> torch.Tensor:
        """音声から特徴抽出 (time_indexは、可変でOK)
        Args:
            x (torch.Tensor): メルスペクトロうグラム (batch_size, n_mels, time_index)

        Returns:
            torch.Tensor: 特徴ベクトル (batch_size, hidden_size)
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat((mu,sg), 1)
        x = self.bn5(x)
        
        mu = self.fc6_mu(x)
        logvar = self.fc6_logvar(x)
        return mu, logvar
    
    def forward(self, x):
        mu, logvar = self.extract_feature(x)
        x = self.sample(mu, logvar)
        x = self.bn6(x)
        if self._use_layer7:
            x = self.fc7(x)
            x = self.bn7(x)
        return x, mu, logvar
    
    @staticmethod
    def kl_loss_of_normal_distribution(mu, logvar, beta=0.5):
        """Returns the Kullback-Leibler divergence loss with a standard Gaussian.

        Args:
            mu (Tensor): Mean of the distribution of shape (B, D, 1).
            logvar (Tensor): Log variance of the distribution of
                shape (B, D, 1).

        Returns:
            Tensor: Kullback-Leibler divergence loss.
        """
        return - beta * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


if __name__ == "__main__":
    spk = ECAPA_TDNN_KL(channel_size=1024)
    x = torch.rand(size=(2, 80, 100))
    x, mu, logvar = spk(x)
    print(x.shape)
    print(mu.shape)
    print(logvar.shape)
    
    print(spk.kl_loss_of_normal_distribution(mu, logvar))