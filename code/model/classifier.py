import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self,
                 per_pixel=False, 
                 embedding_dim=304,
                 leaky_relu=False,
                 out_channels=1,
                 bias=False,
                 hidden_dim=512):
        super().__init__()

        if per_pixel:
            self.layer = nn.Sequential(
                nn.Conv2d(embedding_dim, hidden_dim, kernel_size=1, bias=bias),
                nn.LeakyReLU() if leaky_relu else nn.ReLU(),
                nn.Conv2d(hidden_dim, 256, kernel_size=1, bias=bias),
                nn.LeakyReLU() if leaky_relu else nn.ReLU(),
                nn.Conv2d(256, out_channels, kernel_size=1, bias=False))

        else:
            self.layer = nn.Sequential(                
                nn.Conv2d(embedding_dim, hidden_dim, kernel_size=3, padding=1, bias=bias),
                torch.nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU() if leaky_relu else nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=bias),
                torch.nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU() if leaky_relu else nn.ReLU(),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
            )   
    
    def forward(self, x):
        return self.layer(x)