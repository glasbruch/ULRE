import torch
import torchvision
import numpy as np
import random
import os

from model.hook_network import Network
from model.classifier import Classifier

def set_all_seeds(seed):
    """
    Set all seeds for reproducibility in PyTorch.

    This function sets the seed for PyTorch's random number generator,
    numpy's random number generator, Python's random module, and CUDA
    if it's available.

    Args:
    seed (int): The seed value to use for all random number generators.

    Returns:
    None
    """
    # Set Python's random module seed
    random.seed(seed)

    # Set numpy's random number generator seed
    np.random.seed(seed)

    # Set PyTorch's random number generator seed for CPU
    torch.manual_seed(seed)

    # Set PyTorch's random number generator seed for GPU if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
        
        # These settings ensure deterministic behavior for CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set the seed for any other libraries that might use random numbers
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_model(model, model_file, strict=True):
    state_dict = torch.load(model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']

    state_dict.pop("module.criterion.nll_loss.weight", None)
    model.load_state_dict(state_dict, strict=strict)
    return model

def compute_anomaly_score(bin_logits, mode="probs", activation="exp"):
    if activation == "exp":
        alpha = torch.exp(bin_logits) + 1.0
        # Prevent inf values
        # TODO careful to fix this here
        #max_float = 1e12#torch.finfo(alpha.dtype).max
        #alpha[torch.isinf(alpha)] = max_float
        #alpha = torch.clip(alpha, max=max_float)
    elif activation == "softplus":
        alpha = torch.nn.functional.softplus(bin_logits) + 1.0

    alpha0 = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / alpha0
    
    if mode == "probs":
        #confidence_factor = torch.clamp((alpha0 - 2.0) / (alpha0), min=0.0)
        anomaly_score = probs[:, 1, :, :] #* confidence_factor
    elif mode == "llr":
        #anomaly_score = torch.log(probs[:, 1, :, :]) - torch.log(probs[:, 0, :, :]) 
        anomaly_score = torch.log(alpha[:, 1, :, :]) - torch.log(alpha[:, 0, :, :])
    else:
        raise NotImplementedError
    
    # Gaussian blurring first. Added only after the initial experiments in wandb.
    anomaly_score = torchvision.transforms.GaussianBlur(7, sigma=1)(anomaly_score).squeeze()
    anomaly_score = anomaly_score.cpu().numpy()

    return anomaly_score

def get_anomaly_detector(args, config):
    """
    Get Network Architecture based on arguments provided
    """
    # Initilaize the pretrained model with hooks
    backbone = Network(config.num_classes, args.hook_layers) 
    backbone = load_model(backbone, model_file=config.pretrained_weight_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    backbone.eval()

    model = Classifier(per_pixel=args.per_pixel,
                       embedding_dim=args.embedding_dim,
                       leaky_relu=args.leaky_relu,
                       out_channels=1 if args.loss == "bce" else 2,
                       bias=args.bias,
                       hidden_dim=args.hidden_dim
                       ).to(device)
    
    state_dict = torch.load(args.path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return backbone, model