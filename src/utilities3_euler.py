import torch
import numpy as np
import math

import operator
from functools import reduce


def log_normalize(data, max_ref, min_ref, feature_range=(-1, 1)):
    if isinstance(data, torch.Tensor):
        log_data = torch.log10(data)  # Use torch.log10 for logarithmic scaling
        max_ref, min_ref = np.log10(max_ref), np.log10(min_ref)
        print(f'Using max_ref={max_ref} min_ref={min_ref} to normalize data')
        # min_val, max_val = torch.min(log_data), torch.max(log_data)  # Torch min/max functions
        scaled = (log_data - min_ref) / (max_ref - min_ref)  # Normalize between 0 and 1
        # Scale to the desired feature range
        return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
    else:
        raise TypeError("Input data must be a torch.Tensor")

def maxmin_normalize(data, feature_range=(-1, 1)):
    if isinstance(data, torch.Tensor):
        min_val, max_val = torch.min(data), torch.max(data)  # Use torch min/max functions
        scaled = (data - min_val) / (max_val - min_val)  # Normalize between 0 and 1
        # Scale to the desired feature range
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

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')

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
        print("in abs:",x.shape,y.shape)
        #Assume uniform mesh
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
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

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
        # Ensure predictions are in the range (0, 1)
        y_pred = torch.clamp(y_pred, min=1e-7, max=1-1e-7)
        
        # Calculate binary cross entropy loss manually
        loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        
        if self.reduction:
            if self.size_average:
                return torch.mean(loss)
            else:
                return torch.sum(loss)
        # loss = F.binary_cross_entropy(y_pred.reshape(num_examples,-1), y_true.reshape(num_examples,-1), reduction='mean' if self.size_average else 'sum')

        return loss

    def __call__(self, x, y):
        return self.rel(x, y)
        # return self.binary_cross_entropy(x, y)

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

# check num of tensors and shapes in data lader
def check_dataloader(train_loader):
    for batch in train_loader:
        # print(batch)
        print([x.shape for x in batch])  # If they are tensors, check their shapes
        break

def cosine_similarity(matrix1, matrix2):
    """
    Calculate cosine similarity between two matrices of angles.
    
    :param matrix1: First matrix of angles in radians
    :param matrix2: Second matrix of angles in radians
    :return: Cosine similarity (1 is perfect similarity, -1 is opposite)
    """
    # Convert angles to complex numbers on the unit circle
    complex1 = np.exp(1j * np.radians(matrix1))
    complex2 = np.exp(1j * np.radians(matrix2))
    
    # Calculate the mean of the complex numbers
    mean1 = np.mean(complex1)
    mean2 = np.mean(complex2)
    
    # Calculate cosine similarity
    similarity = np.real(mean1 * np.conj(mean2)) / (np.abs(mean1) * np.abs(mean2))
    
    return similarity

def mean_angular_error(matrix1, matrix2):
    """
    Calculate mean angular error between two matrices of angles.
    
    :param matrix1: First matrix of angles in radians
    :param matrix2: Second matrix of angles in radians
    :return: Mean angular error in radians
    """
    # Calculate the angular difference
    diff = np.radians(matrix1) - np.radians(matrix2)
    
    # Wrap the difference to [-pi, pi]
    wrapped_diff = (diff + np.pi) % (2 * np.pi) - np.pi
    
    # Calculate the mean of the absolute wrapped differences
    mae = np.mean(np.abs(wrapped_diff))
    
    return mae*180/math.pi


def euler_to_orientation_tensor(phi1, Phi, phi2):
    """
    Calculate orientation tensor and its eigenvalues from Euler angles (Bunge convention) using PyTorch tensors.
    
    Parameters:
    phi1, Phi, phi2: torch tensors of Euler angles in degrees
        phi1: first rotation around Z ([-180, 180])
        Phi: rotation around X' ([0, 90])
        phi2: second rotation around Z' ([-180, 180])
    
    Returns:
    eigenvalues: sorted eigenvalues of the orientation tensor (λ1 ≥ λ2 ≥ λ3)
    orientation_tensor: 3x3 orientation tensor
    orientation tensor in convention of:
    xx, xy, xz
    yx, yy, yz
    zx, zy, zz
    """
    # CRSS order must be consistent with A11,A12, A13... order in computing stress tensor!
    # current order is xx,yy,xy
    CRSS = torch.zeros((3, 3), device=phi1.device)
    CRSS[0,0], CRSS[1,1], CRSS[2,2] = 20, 20, 1 #torch.tensor([[20, 0.0, 0.0 ],[0.0, 20, 0.0],[0.0, 0.0, 1.0]]) # [7.0/19*1.7, 0.0, 0.0 ],[0.0, 10.0/19*1.7, 0.0],[0.0, 0.0, 1.0/19*1.7]
    phi1 = phi1[~torch.isnan(phi1)]
    Phi = Phi[~torch.isnan(Phi)]
    phi2 = phi2[~torch.isnan(phi2)]
    # Convert angles from degrees to radians
    phi1_rad = torch.deg2rad(phi1)
    Phi_rad = torch.deg2rad(Phi)
    phi2_rad = torch.deg2rad(phi2)
    
    # Calculate direction cosines (c-axis direction)
    c_axes = torch.zeros((phi1.shape[0], 3), device=phi1.device)
    
    c1, s1 = torch.cos(phi1_rad), torch.sin(phi1_rad)
    c2, s2 = torch.cos(Phi_rad), torch.sin(Phi_rad)
    c3, s3 = torch.cos(phi2_rad), torch.sin(phi2_rad)
    
    # R11 = c1 * c3 - s1 * c2 * s3
    # R12 = -c1 * s3 - s1 * c2 * c3
    R13 = s2 * c1 # s1 * s2
    # R21 = s1 * c3 + c1 * c2 * s3
    # R22 = -s1 * s3 + c1 * c2 * c3
    R23 = s1 * s2 #c1 * s2 
    # R31 = s2 * s3
    # R32 = s2 * c3
    R33 = c2

    # Transform [0, 0, 1] by rotation matrix
    c_axes[:, 0] = R13
    c_axes[:, 1] = R23
    c_axes[:, 2] = R33
    # Calculate second order orientation tensor: average of outer products of c-axis directions
    orientation_tensor = torch.zeros((3, 3), device=phi1.device)
    Rotation = torch.zeros((3,3), device=phi1.device) # rotate by the angle between principle c and z
    N = phi1.shape[0]
    
    for i in range(3):
        for j in range(3):
            orientation_tensor[i, j] = torch.sum(c_axes[:, i] * c_axes[:, j]) / N
    scale = 1/(orientation_tensor[0,0] + orientation_tensor[1,1] + orientation_tensor[2,2])
    orientation_tensor *= scale
    # Calculate eigenvalues
    # first, second, third column of eigenvectors correspond to principle x y z axes
    eigenvalues, eigenvectors = torch.linalg.eigh(orientation_tensor)
    # weakening = torch.mm(torch.mm(eigenvectors,CRSS),eigenvectors.T)
    
    # principle_c_in_xz_cos = eigenvectors[2,0]/torch.sqrt(eigenvectors[0,0]**2+eigenvectors[2,0]**2)
    # principle_c_in_xz = torch.arccos(principle_c_in_xz_cos)
    # if (eigenvectors[0,0]<0 and eigenvectors[2,0]>0):
    #     principle_c_in_xz *= -1
    # elif (eigenvectors[0,0]<0 and eigenvectors[2,0]<0):
    #     principle_c_in_xz = np.pi - principle_c_in_xz
    # Rotation[0,0] =  torch.cos(principle_c_in_xz)**2
    # Rotation[0,1] =  torch.sin(principle_c_in_xz)**2
    # Rotation[0,2] = 2*torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    # Rotation[1,0] =  torch.sin(principle_c_in_xz)**2
    # Rotation[1,1] =  torch.cos(principle_c_in_xz)**2
    # Rotation[1,2] = -2*torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    # Rotation[2,0] = -torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    # Rotation[2,1] =  torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    # Rotation[2,2] =  torch.cos(principle_c_in_xz)**2 - torch.sin(principle_c_in_xz)**2
    # weakening1 = torch.mm(torch.mm(Rotation,CRSS),Rotation.T)
    # # print(f'angle 1: {principle_c_in_xz*180/np.pi}')
    # principle_c_in_xz_cos = eigenvectors[2,1]/torch.sqrt(eigenvectors[0,1]**2+eigenvectors[2,1]**2)
    # principle_c_in_xz = torch.arccos(principle_c_in_xz_cos)
    # if (eigenvectors[0,0]<0 and eigenvectors[2,0]>0):
    #     principle_c_in_xz *= -1
    # elif (eigenvectors[0,0]<0 and eigenvectors[2,0]<0):
    #     principle_c_in_xz = np.pi - principle_c_in_xz
    # Rotation[0,0] =  torch.cos(principle_c_in_xz)**2
    # Rotation[0,1] =  torch.sin(principle_c_in_xz)**2
    # Rotation[0,2] = 2*torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    # Rotation[1,0] =  torch.sin(principle_c_in_xz)**2
    # Rotation[1,1] =  torch.cos(principle_c_in_xz)**2
    # Rotation[1,2] = -2*torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    # Rotation[2,0] = -torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    # Rotation[2,1] =  torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    # Rotation[2,2] =  torch.cos(principle_c_in_xz)**2 - torch.sin(principle_c_in_xz)**2  
    # weakening2 = torch.mm(torch.mm(Rotation,CRSS),Rotation.T)
    # # print(f'angle 2: {principle_c_in_xz*180/np.pi}')
    # principle_c_in_xz_cos = eigenvectors[2,2]/torch.sqrt(eigenvectors[0,2]**2+eigenvectors[2,2]**2)
    # principle_c_in_xz = torch.arccos(principle_c_in_xz_cos)
    # if (eigenvectors[0,0]<0 and eigenvectors[2,0]>0):
    #     principle_c_in_xz *= -1
    # elif (eigenvectors[0,0]<0 and eigenvectors[2,0]<0):
    #     principle_c_in_xz = np.pi - principle_c_in_xz
    # Rotation[0,0] =  torch.cos(principle_c_in_xz)**2
    # Rotation[0,1] =  torch.sin(principle_c_in_xz)**2
    # Rotation[0,2] = 2*torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    # Rotation[1,0] =  torch.sin(principle_c_in_xz)**2
    # Rotation[1,1] =  torch.cos(principle_c_in_xz)**2
    # Rotation[1,2] = -2*torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    # Rotation[2,0] = -torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    # Rotation[2,1] =  torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    # Rotation[2,2] =  torch.cos(principle_c_in_xz)**2 - torch.sin(principle_c_in_xz)**2
    # weakening3 = torch.mm(torch.mm(Rotation,CRSS),Rotation.T)
    # # eigenvalues = eigenvalues.flip(dims=(0,))  # Sort in descending order
    # # scale = np.array([weakening[0,0],weakening[1,1],weakening[2,2]]).min()
    # # weakening[0,:] = weakening[0,:]/weakening[0,0]
    # # weakening[1,:] = weakening[1,:]/weakening[1,1]
    # # weakening[2,:] = weakening[2,:]/weakening[2,2]
    # # print(f'angle 3: {principle_c_in_xz*180/np.pi}')
    v1, v2, v3 = eigenvectors[0,0], eigenvectors[1,0], eigenvectors[2,0]
    d = torch.sqrt(v1**2+v2**2)
    Rotation[0,0] = -v2/d
    Rotation[0,1] = -v1*v3/d
    Rotation[0,2] = v1
    Rotation[1,0] = -v1/d
    Rotation[1,1] = -v2*v3/d
    Rotation[1,2] = v2
    Rotation[2,0] = 0
    Rotation[2,1] = d
    Rotation[2,2] = v3
    weakening1 = torch.mm(torch.mm(Rotation,CRSS),Rotation.T)

    v1, v2, v3 = eigenvectors[0,1], eigenvectors[1,1], eigenvectors[2,1]
    d = torch.sqrt(v1**2+v2**2)
    Rotation[0,0] = -v2/d
    Rotation[0,1] = -v1*v3/d
    Rotation[0,2] = v1
    Rotation[1,0] = -v1/d
    Rotation[1,1] = -v2*v3/d
    Rotation[1,2] = v2
    Rotation[2,0] = 0
    Rotation[2,1] = d
    Rotation[2,2] = v3
    weakening2 = torch.mm(torch.mm(Rotation,CRSS),Rotation.T)

    v1, v2, v3 = eigenvectors[0,2], eigenvectors[1,2], eigenvectors[2,2]
    d = torch.sqrt(v1**2+v2**2)
    Rotation[0,0] = -v2/d
    Rotation[0,1] = -v1*v3/d
    Rotation[0,2] = v1
    Rotation[1,0] = -v1/d
    Rotation[1,1] = -v2*v3/d
    Rotation[1,2] = v2
    Rotation[2,0] = 0
    Rotation[2,1] = d
    Rotation[2,2] = v3
    weakening3 = torch.mm(torch.mm(Rotation,CRSS),Rotation.T)
    
    return eigenvalues.cpu(), 1*(eigenvalues[0]*weakening1 + eigenvalues[1]*weakening2 +  eigenvalues[2]*weakening3).cpu() # 


def f1(lambda_val, n=2):
     return 1 - 3 * (lambda_val - 1/3)**n

def f2(lambda_i, lambda_j, m=1):
    return 1 - torch.abs(lambda_i - lambda_j)**m

def woodcock(lambda_1, lambda_2, lambda_3):
    return torch.sqrt(torch.log(lambda_3/lambda_2)/torch.log(lambda_2/lambda_1))
