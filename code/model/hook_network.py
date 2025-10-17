import torch
import torch.nn as nn
from model.wide_network import DeepWV3Plus
import torchvision

class Hook():
    def __init__(self) -> None:
        self.activation = {}
    
    def get_feature_map(self, name):
        # Hook signature to store input to forward
        def hook(model, input, output):
            # Input is a tuple of length 1
            self.activation[name] = input[0].detach()
        return hook
    
class Network(nn.Module):
    def __init__(self, num_classes, hook_layers):
        super(Network, self).__init__()
        # Subtract additional channel
        self.module = DeepWV3Plus(num_classes)
        self.hook = Hook()

        for name, module in self.module.named_modules():
            if name in hook_layers:
                module.register_forward_hook(self.hook.get_feature_map(name))

    def forward(self, data, output_anomaly=False):
        return self.module(data, output_anomaly=output_anomaly)
 