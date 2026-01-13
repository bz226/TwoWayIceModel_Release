import torch
import numpy as np
import operator
from functools import reduce

#################################################
#
# Utilities
#
#################################################

def log_normalize(data, max_ref, min_ref, feature_range=(-1, 1)):
    if isinstance(data, torch.Tensor):
        log_data = torch.log10(data)  # Use torch.log10 for logarithmic scaling
        max_ref, min_ref = np.log10(max_ref), np.log10(min_ref)
        print(f'Using max_ref={max_ref} min_ref={min_ref} to normalize data')
        # min_val, max_val = torch.min(log_data), torch.max(log_data)  # Torch min/max functions
        scaled = (log_data - min_ref) / (max_ref - min_ref)  # Normalize between 0 and 1
        # Scale to the desired range
        return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
    else:
        raise TypeError("Input data must be a torch.Tensor")

def log_normalize_np(data, max_ref, min_ref, feature_range=(-1, 1)):
    """
    Apply log10 scaling and then min/max normalization to feature_range.
    """
    data = np.log10(data)  # log10 scale
    min_log = np.log10(min_ref)
    max_log = np.log10(max_ref)
    # scale to [0..1]
    data = (data - min_log) / (max_log - min_log)
    # now scale to (feature_range)
    data = data * (feature_range[1] - feature_range[0]) + feature_range[0]
    return data

def maxmin_normalize(data, feature_range=(-1, 1)):
    if isinstance(data, torch.Tensor):
        min_val, max_val = torch.min(data), torch.max(data)  # Use torch min/max functions
        scaled = (data - min_val) / (max_val - min_val)  # Normalize between 0 and 1
        # Scale to the desired range
        return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
    else:
        raise TypeError("Input data must be a torch.Tensor")

def reference_normalize(data, max_ref, min_ref, feature_range=(-1, 1)):
    if isinstance(data, torch.Tensor):
        print(f'Using max_ref={max_ref} min_ref={min_ref} to normalize data')
        scaled = (data - min_ref) / (max_ref - min_ref)  # Normalize between 0 and 1
        # Scale to the desired feature range
        return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
    else:
        raise TypeError("Input data must be a torch.Tensor")

def reference_normalize_np(data, min_ref, max_ref, feature_range=(-1, 1)):
    """
    Apply min/max normalization to feature_range.
    """
    data = (data - min_ref) / (max_ref - min_ref)  # [0..1]
    data = data * (feature_range[1] - feature_range[0]) + feature_range[0]
    return data

def smoothness_loss(predictions):
    # prediction should be in shapes [num batch, kde reso]
    # compute temporal derivatives using finite differences
    temporal_diff = predictions[:, 1:] - predictions[:, :-1]
    # penalize large changes between consecutive time steps
    smoothness_penalty = torch.sum(temporal_diff**2)
    return smoothness_penalty

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        #assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        # compute the norm of order self.p, 1 meaning norm is calculated across the second dimension 
        # (the flattened feature dimension) for each example in the batch
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms)
            else:
                return torch.sum(diff_norms)

        return diff_norms

    # added bross entropy loss for binary prediction
    def binary_cross_entropy(self, y_pred, y_true):
        """
        Calculate binary cross entropy loss.
        Args:
            y_pred (torch.Tensor): Predicted probabilities.
            y_true (torch.Tensor): Ground truth labels.
        Returns:
            torch.Tensor: The binary cross entropy loss.
        """
        num_examples = y_pred.size()[0]
        y_pred,y_true = y_pred.reshape(num_examples,-1),y_true.reshape(num_examples,-1)
        # ensure predictions are in the range (0, 1)
        y_pred = torch.clamp(y_pred, min=1e-7, max=1-1e-7)
        
        # calculate binary cross entropy loss manually
        loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        
        if self.reduction:
            if self.size_average:
                return torch.mean(loss)
            else:
                return torch.sum(loss)
        # loss = F.binary_cross_entropy(y_pred.reshape(num_examples,-1), y_true.reshape(num_examples,-1), reduction='mean' if self.size_average else 'sum')

        return loss

    def __call__(self, x, y):
        # here defines what loss function to use
        return self.rel(x, y)
        # return self.binary_cross_entropy(x, y)

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

def mem_needed_bytes(total_samples, grid_size, step_known, step_predict, dtype_bytes=4):
    total_elements = total_samples * grid_size * (step_predict + 4*step_known)
    return dtype_bytes * total_elements

def human(nbytes):
    for unit in ["B","KiB","MiB","GiB","TiB"]:
        if nbytes < 1024 or unit == "TiB":
            return f"{nbytes:,.2f} {unit}"
        nbytes /= 1024