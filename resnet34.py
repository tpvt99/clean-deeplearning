#%%
import os
import sys
import torch as t
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import einops
from dataclasses import dataclass
import torchvision
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from typing import List, Tuple, Dict, Type
from PIL import Image
from IPython.display import display
import json
import pandas as pd
from jaxtyping import Float, Int
from pytorch_lightning.loggers import CSVLogger
import numpy as np

# Make sure exercises are in the path
from torchinfo import summary
import torchview
import wandb

from utils import print_param_count

device = torch.device(f"cuda:{t.cuda.current_device()}")  \
    if t.cuda.is_available() else torch.device('cpu')

print(device)

debug = True # Adjust this to False if not printing to understand the models

MAIN = __name__ == "__main__"
# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride

        # In the paper, residual block are 2 blocks sit inside the skip conv (no dashed), 
        # e.g. for 34-residual
        #
        # 3x3 conv, 64
        # 3x3 conv, 64
        #
        # Input has shape (b, 64, 56, 56), thus because first_stride = 1(no dashed skip connection),
        # padding must be 1 so that output side is still (b, 64, 56, 56)

        ## However, if first_stride > 1 (dashed skip connection), inputs and outputs are:
        # Input shape (b, 64, 56, 56), output shape is (b, 128, 28, 28)
        # which is doubling the channels and downsampling the height and width
        # thus in main branch, we use stride=first_strde to reduce the height and width
        # and the same time, we add skip_branch with stride=first_stride to also reduce height and width
        # we keep kernel = 3, padding = 1 in both stride=1 and stride > 1

        
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_feats, out_channels=out_feats, kernel_size=3, stride=first_stride,
                   padding = 1, bias=False),
            nn.BatchNorm2d(num_features=out_feats),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_feats, out_channels=out_feats, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_feats)
        )

        self.skip_branch = nn.Identity()

        if first_stride > 1:
            self.skip_branch = nn.Sequential(
                nn.Conv2d(in_channels=in_feats, out_channels=out_feats, kernel_size=1, stride=first_stride,
                       padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_feats)
            )


        self.relu = nn.ReLU()

    def forward(self, x: Float[Tensor, "batch c h w"]) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        '''

        #residual = self.main_branch(x)
        #residual = residual + self.skip_branch(x)

        residual = self.skip_branch(x)
        residual = residual + self.main_branch(x)

        output = self.relu(residual)

        return output 
    
    def extra_repr(self) -> str:
        return f"in_feats {self.in_feats} out_feats {self.out_feats} first_stride {self.first_stride}"
 
# %%
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride: int = 1):
        '''
            An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.
        '''
        super().__init__()

        # In the paper we can see that the first block never has first_strde > 1
        
        ## Can use either Sequential or ModuleList
        residual_blocks = []
        residual_blocks.append(ResidualBlock(in_feats=in_feats, 
                                                 out_feats=out_feats,
                                                 first_stride=first_stride))
        
        for _ in range(1, n_blocks):
            residual_blocks.append(ResidualBlock(in_feats=out_feats,
                                                 out_feats=out_feats,
                                                 first_stride=1))

        # it blocks my view, should not use sequentials 
        self.module_sequential = nn.Sequential(*residual_blocks)


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        
        # If sequential:
        
        output = self.module_sequential(x)

        return output
# %%
class ResNet34(nn.Module):
    def __init__(
            self,
            n_blocks_per_group = (3, 4, 6, 3),
            in_features_per_group = (64, 64, 128, 256),
            out_features_per_group = (64, 128, 256, 512),
            first_strides_per_group = (1, 2, 2, 2),
            n_classes = 1000
    ):
        super().__init__()
        self.n_blocks_per_group = n_blocks_per_group
        self.in_features_per_group = in_features_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        # conv1 with 7x7 kernel, 64 channels, stride=2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features = 64)
        self.relu1 = nn.ReLU()

        # conv2_pool with 3x3 maxpool, stride=2
        self.conv2_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # conv2, conv3, conv4, conv5
        blocks = []
        for i in range(len(n_blocks_per_group)):
            blocks.append(BlockGroup(n_blocks = n_blocks_per_group[i],
                                          in_feats = in_features_per_group[i],
                                          out_feats = out_features_per_group[i],
                                          first_stride = first_strides_per_group[i]))

        # This blocks view of summary 
        self.blocks = nn.Sequential(*blocks)

        # average pool
        self.avg_pool = t.nn.AvgPool2d(kernel_size=7)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(in_features = 512, out_features=1000)


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
            x: shape (batch, channels, height, width)
            Return: shape (batch, n_classes)
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2_maxpool(x)
        #for index, module in enumerate(self.blocks):
            #x = module(x)
        
        x = self.blocks(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x

my_resnet = ResNet34()


# %%
# Veryfing your implementation by loading weights
def copy_weights(my_resnet : ResNet34, pretrained_resnet: torchvision.models.resnet34, copy=False) -> ResNet34:
    '''
        Copy over the weights of `pretrained_resnet` to your resnet
    '''
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Missing state dictionaries"

    # Define a dictionary mapping the names of your parameters/buffer to their values in pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())
    }

    # Load in this dictionary to your model
    if copy:
        my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet

 #%%
 ## This is to print how many parameters
if MAIN and debug:
    pretrained_resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    print(summary(my_resnet, input_data = torch.rand(2, 3, 224, 224), verbose=0, depth=10))
    print(summary(pretrained_resnet, input_data = torch.rand(2,3,224,224)))
#%%
## This is to load pre-trained weights
if MAIN and debug:
 
    pretrained_resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    my_resnet = copy_weights(my_resnet, pretrained_resnet)
# %%
if MAIN and debug:
    print_param_count(my_resnet, pretrained_resnet)
# %%
