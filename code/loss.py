import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.special import digamma, gammaln

class EDLLoss(nn.Module):
    def __init__(self, 
                 cor_reg=False,
                 log_loss=False,
                 num_classes=2,
                 activation = "exp",
                 annealing_param=10.0,
                 precision_reg=0,
                 beta=1
        ):
        super(EDLLoss, self).__init__()
        self.cor_reg = cor_reg
        self.log_loss = log_loss
        self.num_classes = num_classes
        self.activation = activation
        self.annealing_param = annealing_param
        self.precision_reg = precision_reg
        self.beta = beta
    
    def kl_divergence(self, alpha, num_classes, device=None):
        if not device:
            device = alpha.device
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl
    
    def reverse_kl_divergence(self, alpha, num_classes, device=None):
        if device is None:
            device = alpha.device
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)

        # KL(Dir(1) || Dir(alpha))
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (ones - alpha)
            .mul(torch.digamma(ones) - torch.digamma(ones.sum(dim=1, keepdim=True)))
            .sum(dim=1, keepdim=True)
        )
        reverse_kl = first_term + second_term
        return reverse_kl
    
    def entropy(self, alpha, num_classes):
        # Differential Entropy
        """
        #sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        sum_alpha = torch.sum(alpha, dim=1)

        #print(f"sum: {sum_alpha.shape}")
        log_B = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(sum_alpha)
        #print(f"log b: {log_B.shape}")
        #entropy = log_B + (sum_alpha - num_classes) * torch.digamma(sum_alpha) - ((alpha - 1) * torch.digamma(alpha)).sum(dim=-1)
        digamma_alpha = torch.digamma(alpha)
        #digamma_alpha0 = torch.digamma(sum_alpha).unsqueeze(-1)

        # Compute (alpha - 1) * digamma(alpha) efficiently
        second_term = (alpha - 1).mul(digamma_alpha).sum(dim=-1)

        entropy = log_B + (sum_alpha - num_classes) * torch.digamma(sum_alpha) - second_term

        return entropy"""
        # Calculate the sum of alpha for each parameter vector in the batch
        alpha_sum = alpha.sum(dim=1, keepdim=True)  # [batch_size, 1]
    
        # Calculate log of the multivariate beta function
        # log B(α) = sum_i log(Γ(α_i)) - log(Γ(sum_i α_i))
        log_beta = gammaln(alpha).sum(dim=1) - gammaln(alpha_sum.squeeze())
    
        # Calculate the term (α_i - 1) * ψ(α_i) summed over i
        digamma_term1 = ((alpha - 1) * digamma(alpha)).sum(dim=1)
    
        # Calculate the term (sum_i α_i - K) * ψ(sum_i α_i)
        # where K is the dimension of alpha
        k = alpha.size(1)
        digamma_term2 = (alpha_sum.squeeze() - k) * digamma(alpha_sum.squeeze())
    
        # Calculate the entropy
        entropy = log_beta + digamma_term1 - digamma_term2
        return entropy
    
    def forward(self, logits, target, epoch):
        target = target.flatten()
        void_mask = target == 255
        ood_mask = target == 254

        target[~ood_mask] = 0
        target[ood_mask] = 1
        target = target[~void_mask]
        target_1hot = F.one_hot(target, num_classes=self.num_classes).float()

        B, C, H, W = logits.shape
        logits = logits.reshape(B, C, -1) # B x C x HW
        logits = logits.permute(0, 2, 1) # B x HW x C
        logits = logits.reshape(-1, C) # BHW X C

        # Remove ignore index
        logits = logits[~void_mask]

        # Undersample ID
        """
        ood_mask = target == 1
        id_mask = torch.empty_like(ood_mask)

        num_ood = len(logits[ood_mask])
        num_id = len(logits[~ood_mask])
        idxs = torch.randperm(num_id)[:num_ood]
        mask = torch.zeros_like(ood_mask)
        mask[ood_mask] = True
        mask[idxs] = True
        logits = logits[mask]
        target = target[mask]
        target_1hot = target_1hot[mask]"""

        if self.activation == "exp":
            alpha = torch.exp(logits) + 1.0
            # Prevent inf values
            #max_float = 1e12#torch.finfo(alpha.dtype).max
            #alpha[torch.isinf(alpha)] = max_float
            #alpha = torch.clip(alpha, max=max_float)
            
            #alpha = torch.clip(alpha, max=100)
        elif self.activation == "softplus":
            alpha = torch.nn.functional.softplus(logits) + 1.0

        print(f"Alpha0: {alpha[:,0].min()} {alpha[:,0].max()}, mean: {alpha[:,0].mean()}")
        print(f"Alpha1: {alpha[:,1].min()} {alpha[:,1].max()}, mean: {alpha[:,1].mean()}")

        S = alpha.sum(dim=1, keepdim=True)

        # Evidential log loss
        if self.log_loss:
            loss = torch.sum(target_1hot * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True) # KLD expects shape Nx1
            #loss = torch.sum(target_1hot * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=False)
        else:
            # MSE loss
            probs = alpha/S
            error = (target_1hot - probs)**2
            variance = probs * (1.0 - probs)/(S + 1)
            loss = torch.sum(error + variance, dim=1, keepdim=True)

        # KLD Regularization
        annealing_coef = np.minimum(1.0, epoch / self.annealing_param)
        kl_alpha = (alpha - 1) * (1 - target_1hot) + 1
        # Forward KLD
        kl_div = annealing_coef * self.kl_divergence(kl_alpha, self.num_classes)
        # Reverse KLD
        #kl_div = annealing_coef * self.reverse_kl_divergence(kl_alpha, self.num_classes)

        loss += self.beta * kl_div

        # Entropy Regularization
        #entropy = self.entropy(alpha, self.num_classes)
        #loss -= self.beta * entropy

        return loss.mean() #+ self.precision_reg * torch.mean(torch.log(S))
    
        """if self.cor_reg:
            with torch.no_grad():
                vacuity = self.num_classes / S.detach()
            
            cor_reg = vacuity * target_1hot * logits
            cor_reg = cor_reg.sum() / cor_reg.shape[0]
        else:
            cor_reg = 0

        #print(f"Correct reg: {cor_reg}")
        return loss.mean() - cor_reg"""

class IEDLLoss(nn.Module):
    def __init__(self, 
                 cor_reg=False,
                 log_loss=False,
                 num_classes=2,
                 activation = "exp",
                 annealing_param=10.0,
                 precision_reg=0,
                 beta=1
        ):
        super(IEDLLoss, self).__init__()
        self.cor_reg = cor_reg
        self.log_loss = log_loss
        self.num_classes = num_classes
        self.activation = activation
        self.annealing_param = annealing_param
        self.precision_reg = precision_reg
        self.beta = beta
    
    def kl_divergence(self, alpha, num_classes, device=None):
        if not device:
            device = alpha.device
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl
    
    def reverse_kl_divergence(self, alpha, num_classes, device=None):
        if device is None:
            device = alpha.device
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)

        # KL(Dir(1) || Dir(alpha))
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (ones - alpha)
            .mul(torch.digamma(ones) - torch.digamma(ones.sum(dim=1, keepdim=True)))
            .sum(dim=1, keepdim=True)
        )
        reverse_kl = first_term + second_term
        return reverse_kl
    
    def compute_fisher_loss(self, labels_1hot, alpha):
        # From: https://github.com/danruod/IEDL/blob/main/code_fsl/train.py#L10

        # batch_dim, n_samps, num_classes = evi_alp_.shape
        evi_alp0_ = torch.sum(alpha, dim=-1, keepdim=True)

        gamma1_alp = torch.polygamma(1, alpha)
        gamma1_alp0 = torch.polygamma(1, evi_alp0_)

        print(f"gamma1_alp: {torch.any(torch.isnan(gamma1_alp))} gamma1_alp0:  {torch.any(torch.isnan(gamma1_alp0))}")
        gap = labels_1hot - alpha / evi_alp0_

        loss_mse_ = (gap.pow(2) * gamma1_alp).sum(-1)

        loss_var_ = (alpha * (evi_alp0_ - alpha) * gamma1_alp / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))).sum(-1)

        loss_det_fisher_ = - (torch.log(gamma1_alp).sum(-1) + torch.log(1.0 - (gamma1_alp0 / gamma1_alp).sum(-1)))
        loss_det_fisher_ = torch.where(torch.isfinite(loss_det_fisher_), loss_det_fisher_, torch.zeros_like(loss_det_fisher_))

        return loss_mse_.mean(), loss_var_.mean(), loss_det_fisher_.mean()

    def forward(self, logits, target, epoch):
        target = target.flatten()
        void_mask = target == 255
        ood_mask = target == 254

        target[~ood_mask] = 0
        target[ood_mask] = 1
        target = target[~void_mask]
        target_1hot = F.one_hot(target, num_classes=self.num_classes).float()

        B, C, H, W = logits.shape
        logits = logits.reshape(B, C, -1) # B x C x HW
        logits = logits.permute(0, 2, 1) # B x HW x C
        logits = logits.reshape(-1, C) # BHW X C

        # Remove ignore index
        logits = logits[~void_mask]

        print(f"Logits0: {logits[:,0].min()} {logits[:,0].max()}, mean: {logits[:,0].mean()}")
        print(f"Logits1: {logits[:,1].min()} {logits[:,1].max()}, mean: {logits[:,1].mean()}")

        if self.activation == "exp":
            alpha = torch.exp(logits) + 1.0

        elif self.activation == "softplus":
            alpha = torch.nn.functional.softplus(logits) + 1.0

        print(f"Alpha0: {alpha[:,0].min()} {alpha[:,0].max()}, mean: {alpha[:,0].mean()}")
        print(f"Alpha1: {alpha[:,1].min()} {alpha[:,1].max()}, mean: {alpha[:,1].mean()}")

        mean, var, fisher_det = self.compute_fisher_loss(target_1hot, alpha)
        print(f"Mean: {mean} {torch.any(torch.isnan(mean))} Var: {var} {torch.any(torch.isnan(var))} Fisher: {fisher_det} {torch.any(torch.isnan(fisher_det))}")

        # KLD Regularization
        kl_alpha = (alpha - 1) * (1 - target_1hot) + 1
        # Forward KLD
        lambda_kl = np.minimum(1.0, epoch / self.annealing_param)
        kl_alpha = (alpha - 1) * (1 - target_1hot) + 1
        kl_div = self.kl_divergence(kl_alpha, self.num_classes).mean()
        
        lambda_fisher = 0.1
        loss = mean + var #+ lambda_fisher * fisher_det + lambda_kl * kl_div
        print(f"Loss: {loss.shape} {loss.item()}")

        return loss#.mean() 
          
class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, target):
        target = target.flatten()
        void_mask = target == 255
        ood_mask = target == 254

        target[~ood_mask] = 0
        target[ood_mask] = 1
        target = target[~void_mask].float()
        #target_1hot = F.one_hot(target, num_classes=self.num_classes).float()

        B, C, H, W = logits.shape
        logits = logits.reshape(B, C, -1) # B x C x HW
        logits = logits.permute(0, 2, 1) # B x HW x C
        logits = logits.reshape(-1) # BHW

        # Remove ignore index
        logits = logits[~void_mask]

        loss = self.bce(logits, target)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        #self.bce = nn.BCEWithLogitsLoss()
        self.gamma = gamma
    
    def forward(self, logits, target):
        target = target.flatten()
        void_mask = target == 255
        ood_mask = target == 254

        target[~ood_mask] = 0
        target[ood_mask] = 1
        target = target[~void_mask].float()
        #target_1hot = F.one_hot(target, num_classes=self.num_classes).float()

        B, C, H, W = logits.shape
        logits = logits.reshape(B, C, -1) # B x C x HW
        logits = logits.permute(0, 2, 1) # B x HW x C
        logits = logits.reshape(-1) # BHW

        # Remove ignore index
        logits = logits[~void_mask]

        log_probs = F.logsigmoid(logits) * target + F.logsigmoid(-logits) * (1 - target)
        probs = torch.exp(log_probs)

        focal_term = (1 - probs) ** self.gamma

        loss = -focal_term * log_probs
        #loss = self.bce(logits, target)

        return loss.mean()

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, logits, target):
        target = target.flatten()
        void_mask = target == 255
        ood_mask = target == 254

        target[~ood_mask] = 0
        target[ood_mask] = 1
        target = target[~void_mask]

        B, C, H, W = logits.shape
        logits = logits.reshape(B, C, -1) # B x C x HW
        logits = logits.permute(0, 2, 1) # B x HW x C
        logits = logits.reshape(-1, C) # BHW X C

        # Remove ignore index
        logits = logits[~void_mask]

        loss = self.ce(logits, target)
        return loss
    
class HSCLoss(nn.Module):
    def __init__(self, hsc_norm):
        super(HSCLoss, self).__init__()
        self.hsc_norm = hsc_norm
        self.eps = 1e-9

    def forward(self, inputs, targets):
        # Void mask of pixel to ignore e.g., Motorhaube
        void_mask = targets == 255
        ood_mask = targets == 254

        targets[~ood_mask] = 0
        targets[ood_mask] = 1

        if self.hsc_norm == 'l1':
            dists = torch.norm(inputs, p=1, dim=1)
        if self.hsc_norm == 'l2':
            dists = torch.norm(inputs, p=2, dim=1)
        if self.hsc_norm == 'l2_squared':
            dists = torch.norm(inputs, p=2, dim=1) ** 2
        if self.hsc_norm == 'l2_squared_linear':
            dists = torch.sqrt(torch.norm(inputs, p=2, dim=1) ** 2 + 1) - 1

        scores = 1 - torch.exp(-dists)
        losses = torch.where(targets == 0, dists, -torch.log(scores + self.eps))
        loss = torch.mean(losses[~void_mask])
        return loss#, scores

class DiceLoss(nn.Module):
    """
    DICE Loss for binary segmentation with support for ignore indices.
    
    The DICE coefficient measures the overlap between two binary masks:
    DICE = 2 * |A ∩ B| / (|A| + |B|)
    
    DICE Loss = 1 - DICE coefficient
    """
    
    def __init__(self, smooth=1e-6, reduction='mean', ignore_index=255):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero
            reduction (str): Specifies the reduction to apply to the output:
                           'none' | 'mean' | 'sum'
            ignore_index (int, list, or None): Index or list of indices to ignore.
                                              Pixels with these values in targets will be ignored.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        # Convert single ignore index to list for uniform handling
        if self.ignore_index is not None:
            if isinstance(self.ignore_index, int):
                self.ignore_index = [self.ignore_index]
            elif not isinstance(self.ignore_index, list):
                self.ignore_index = list(self.ignore_index)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (N, 1, H, W) or (N, H, W)
                                 Can be logits or probabilities
            targets (torch.Tensor): Ground truth of shape (N, 1, H, W) or (N, H, W)
                                  Should contain binary values (0 or 1) and ignore indices
        
        Returns:
            torch.Tensor: DICE loss
        """
        # Ensure inputs are probabilities (apply sigmoid if they appear to be logits)
        if inputs.min() < 0 or inputs.max() > 1:
            inputs = torch.sigmoid(inputs)
        
        # Create mask for pixels to include (not ignore)
        if self.ignore_index is not None:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
            for ignore_idx in self.ignore_index:
                valid_mask = valid_mask & (targets != ignore_idx)
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
        
        # Apply mask to inputs and targets
        inputs_masked = inputs * valid_mask.float()
        targets_masked = targets * valid_mask.float()
        
        # Flatten the tensors to compute intersection and union
        inputs_flat = inputs_masked.view(inputs_masked.size(0), -1)  # (N, H*W)
        targets_flat = targets_masked.view(targets_masked.size(0), -1)  # (N, H*W)
        valid_flat = valid_mask.view(valid_mask.size(0), -1).float()  # (N, H*W)
        
        # Compute intersection and union only for valid pixels
        intersection = (inputs_flat * targets_flat).sum(dim=1)  # (N,)
        union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)  # (N,)
        
        # Handle edge case where no valid pixels exist
        valid_pixel_count = valid_flat.sum(dim=1)  # (N,)
        
        # For samples with no valid pixels, set DICE to 1 (loss = 0)
        dice_coeff = torch.where(
            valid_pixel_count > 0,
            (2.0 * intersection + self.smooth) / (union + self.smooth),
            torch.ones_like(intersection)
        )
        
        # DICE loss is 1 - DICE coefficient
        dice_loss = 1.0 - dice_coeff
        
        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:  # 'none'
            return dice_loss

'''
class DiceLoss(nn.Module):
    """
    DICE Loss for binary segmentation.
    
    The DICE coefficient measures the overlap between two binary masks:
    DICE = 2 * |A ∩ B| / (|A| + |B|)
    
    DICE Loss = 1 - DICE coefficient
    """
    
    def __init__(self, smooth=1e-6, reduction='mean'):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero
            reduction (str): Specifies the reduction to apply to the output:
                           'none' | 'mean' | 'sum'
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (N, 1, H, W) or (N, H, W)
                                 Can be logits or probabilities
            targets (torch.Tensor): Ground truth of shape (N, 1, H, W) or (N, H, W)
                                  Should contain binary values (0 or 1)
        
        Returns:
            torch.Tensor: DICE loss
        """
        # Ensure inputs are probabilities (apply sigmoid if they appear to be logits)
        if inputs.min() < 0 or inputs.max() > 1:
            inputs = torch.sigmoid(inputs)
        
        # Flatten the tensors to compute intersection and union
        inputs_flat = inputs.view(inputs.size(0), -1)  # (N, H*W)
        targets_flat = targets.view(targets.size(0), -1)  # (N, H*W)
        
        # Compute intersection and union
        intersection = (inputs_flat * targets_flat).sum(dim=1)  # (N,)
        union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)  # (N,)
        
        # Compute DICE coefficient
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # DICE loss is 1 - DICE coefficient
        dice_loss = 1.0 - dice_coeff
        
        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:  # 'none'
            return dice_loss

            '''

if __name__ == "__main__":
    criterion = EDLLoss()
    alpha = torch.tensor([
        [0.5, 0.5, 0.5],  # Sparse symmetric Dirichlet
        [5.0, 5.0, 5.0],  # Concentrated symmetric Dirichlet
        [1.0, 2.0, 3.0]   # Asymmetric Dirichlet
    ])
    print(f"Alpha: {alpha.shape}")
    entropy = criterion.entropy(alpha, 3)
    print(entropy)