""" This code is based on Zhang, T., Trad, D., & Innanen, K. (2023). 
Learning to solve the elastic wave equation with Fourier neural operators. 
Geophysics, 88(3), T101-T119.

Modified by: Emma Liu
Email: liuwj@stanford.edu
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utilities3_grain import *


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.

        input: batchsize x channel x x_grid x y_grid (channel means the number of input channels)
        output: batchsize x channel x x_grid x y_grid

        Parameters:
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        modes1: int
            Number of Fourier modes in x direction.
        modes2: int
            Number of Fourier modes in y direction.
        """

        super(SpectralConv2d_fast, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1/(self.in_channels*self.out_channels*1000)) 
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, \
                                                             self.modes1, self.modes2, dtype=torch.complex64))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, \
                                                             self.modes1, self.modes2, dtype=torch.complex64))

        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, \
                                                             self.modes1, self.modes2, dtype=torch.complex64))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, \
                                                             self.modes1, self.modes2, dtype=torch.complex64))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        """ (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y) """
        return torch.einsum("bixy,ioxy->boxy", input, weights)  # Regular 4D case
            # return torch.einsum("bixy,ioxy->boxy", input, weights)
        
    def forward(self, x):
        """ Multiply relevant Fourier modes and return to physical space
        """
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft_x = torch.fft.rfft2(x)
        x_ft_z = torch.fft.rfft2(torch.transpose(x,2,3))

        # Multiply relevant Fourier modes
        out_ft_x = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, \
                             dtype=torch.complex64, device=x.device)
        out_ft_z = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, \
                             dtype=torch.complex64, device=x.device)
        out_ft_x[:, :, :self.modes1, :self.modes2]  = self.compl_mul2d(x_ft_x[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft_x[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft_x[:, :, -self.modes1:, :self.modes2], self.weights2)


        out_ft_z[:, :, :self.modes1, :self.modes2]  = self.compl_mul2d(x_ft_z[:, :, :self.modes1, :self.modes2], self.weights3)
        out_ft_z[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft_z[:, :, -self.modes1:, :self.modes2], self.weights4)

        #Return to physical space
        out_ft_x = torch.fft.irfft2(out_ft_x, s=(x.size(-2), x.size(-1)))
        out_ft_z = torch.fft.irfft2(out_ft_z, s=(x.size(-2), x.size(-1)))
        
        return out_ft_x, out_ft_z
    


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, step_known, activation_func,loss_func):
        """
        Inherits from nn.Module PyTorch
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        modes 1 & 2: int
            number of fourier modes (freq) in x and y direction when doing FFT
        width: int
            number of channels in hidden layers in NN
        """

        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.activation_func = activation_func
        self.loss_func = loss_func

        #  lifts the input data from an input channel dimension of 4 (grainszie, strain, temp, pressure)
        #  to the specified width. 5 is. 5 is 
        # [input x (distribution of grain size) * num_known_step , num x cells, C1* num_known_step, C2* num_known_step, C3* num_known_step]
        # how to commpute: # = 4 * num_known_step + 2 (2 is 2D grid, 3 if 3D, 1 if 1D)
        self.fc0 = nn.Linear(4 * step_known + 2, self.width)
        
        if activation_func == 'sig':
            self.activation = nn.Sigmoid()
        elif activation_func == 'relu':
            self.activation = nn.ReLU()
        elif activation_func == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unrecognized activation function. Choose 'sig', 'relu', or 'tanh'.")

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv6 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4 = nn.Conv2d(self.width, self.width, 1)
        self.w5 = nn.Conv2d(self.width, self.width, 1)
        self.w6 = nn.Conv2d(self.width, self.width, 1)

        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)
        self.bn4 = torch.nn.BatchNorm2d(self.width)
        self.bn5 = torch.nn.BatchNorm2d(self.width)
        self.bn6 = torch.nn.BatchNorm2d(self.width)

        # self.bn0 = torch.nn.InstanceNorm2d(self.width, affine=True, track_running_stats=False)
        # self.bn1 = torch.nn.InstanceNorm2d(self.width, affine=True, track_running_stats=False)
        # self.bn2 = torch.nn.InstanceNorm2d(self.width, affine=True, track_running_stats=False)
        # self.bn3 = torch.nn.InstanceNorm2d(self.width, affine=True, track_running_stats=False)
        # self.bn4 = torch.nn.InstanceNorm2d(self.width, affine=True, track_running_stats=False)
        # self.bn5 = torch.nn.InstanceNorm2d(self.width, affine=True, track_running_stats=False)
        # self.bn6 = torch.nn.InstanceNorm2d(self.width, affine=True, track_running_stats=False)

        self.fc1 = nn.Linear(self.width,128)
        self.fc2 = nn.Linear(128,1)

    def to(self, device):
        # Move the model to the specified device
        model = super().to(device)
        # Handle complex parameters
        for name, param in model.named_parameters():
            if param.dtype == torch.complex64 or param.dtype == torch.complex128:
                param.data = param.data.to(device)
        return model
    
    def forward(self, x , C1, C2, C3):
        """ The forward propagation of neural network
        """

        batchsize = x.shape[0]
        
        size_x, size_y,num_know_steps = x.shape[1], x.shape[2], x.shape[3]
        C1 = torch.reshape(C1,[batchsize,num_know_steps,size_x,size_y])
        C2 = torch.reshape(C2,[batchsize,num_know_steps,size_x,size_y])
        C3 = torch.reshape(C3,[batchsize,num_know_steps,size_x,size_y])

        C1 = torch.nn.functional.normalize(C1)
        C2 = torch.nn.functional.normalize(C2)
        C3 = torch.nn.functional.normalize(C3)

        C1 = torch.reshape(C1,[batchsize,size_x,size_y,num_know_steps])
        C2 = torch.reshape(C2,[batchsize,size_x,size_y,num_know_steps])
        C3 = torch.reshape(C3,[batchsize,size_x,size_y,num_know_steps])

        # grid is of size: [batch size, num x cells, num y cells, 2] 2 means 2D
        grid = self.get_grid(batchsize, size_x, size_y, x.device)
        # x after concatenation: [num batch, num x cells, num y cells, 6] 6 is [x, grid x, grid y, c1, c2, c3]
        x = torch.cat((x, grid, C1, C2,C3), dim=-1) # concatenant the x y coordinates into training data
        
        # del grid, C1, C2,C3
        x = self.fc0(x)
        # after fc0, x has shape [num batch, num x cells, num y cells, num width (set in train)]
        x = x.permute(0, 3, 1, 2) # [num batch, num width (set in train), num x cells, num y cells]
        x0, z0 = self.conv0(x)
        xw_0   = self.w0(x)
        x      = self.bn0(xw_0 + x0 + z0 )
        x = self.activation(x)
        
        x1, z1 = self.conv1(x)
        xw_1   = self.w1(x)
        x      = self.bn1(xw_1 + x1 + z1 + x)
        x = self.activation(x)

        x2, z2 = self.conv2(x)
        xw_2   = self.w2(x)
        x      = self.bn2(xw_2 + x2 + z2 + x)
        x = self.activation(x)
        
        
        x3, z3 = self.conv3(x)
        xw_3   = self.w3(x)
        x      = self.bn3(xw_3 + x3 + z3 + x)
        x = self.activation(x)

        # x4, z4 = self.conv4(x)
        # xw_4   = self.w4(x)
        # x      = self.bn4(xw_4 + x4 + z4 + x)
        # x = self.activation(x)

        # x5, z5 = self.conv5(x)
        # xw_5   = self.w5(x)
        # x      = self.bn5(xw_5 + x5 + z5 + x)
        # x = self.activation(x)

        # x6, z6 = self.conv6(x)
        # xw_6   = self.w6(x)
        # x      = self.bn6(xw_6 + x6 + z6 + x)
        # x = self.activation(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)
        return x

    def get_grid(self, batchsize, size_x, size_y, device):
        """ Generate coordinate grid scaled to interval [-1,1]
        """

        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        
        return torch.cat((gridx, gridy), dim=-1).to(device)