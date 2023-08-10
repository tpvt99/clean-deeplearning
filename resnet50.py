#%%
import os
import sys
import torch as t
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from dataclasses import dataclass
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from typing import List, Tuple, Dict, Type, Optional
from PIL import Image
from IPython.display import display
import pandas as pd
from jaxtyping import Float, Int
import numpy as np
import time

# Library for viewing networks and assert they same to original network
from torchinfo import summary
from utils import print_param_count
# For logging to wandb
import wandb

device = torch.device(f"cuda:{t.cuda.current_device()}")  \
    if t.cuda.is_available() else torch.device('cpu')

print(device)

debug = False # Adjust this to False if not printing to understand the models

MAIN = __name__ == "__main__"
# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, inner_feats:int,  out_feats: int, first_stride=1):
        '''
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()
        self.in_feats = in_feats
        self.inner_feats = inner_feats
        self.out_feats = out_feats
        self.first_stride = first_stride

        self.main_branch = nn.Sequential(
                nn.Conv2d(in_channels=in_feats, out_channels=inner_feats, kernel_size=1, stride=first_stride,
                            padding = 0, bias=False),
                nn.BatchNorm2d(num_features=inner_feats),
                nn.ReLU(),
                nn.Conv2d(in_channels=inner_feats, out_channels=inner_feats, kernel_size=3, stride=1, padding=1,
                            bias=False),
                nn.BatchNorm2d(num_features=inner_feats),
                nn.ReLU(),
                nn.Conv2d(in_channels=inner_feats, out_channels=out_feats, kernel_size=1, stride=1, padding=0,
                            bias=False),
                nn.BatchNorm2d(num_features=out_feats)
        )
        

        self.skip_branch = nn.Identity()

        # first_stride > 1 for changing width/height for blocks from 2->3, 3->4, 4->5
        # it is enough for resnet34 because each block does not change the channels
        # for resnet50, 101, 152, we need to add condition is that if out_c != in_c, then we also add skip
        if first_stride > 1 or out_feats != in_feats:
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
        residual = self.main_branch(x)

        skip = self.skip_branch(x)

        output = residual + skip

        output = self.relu(output)

        return output 
    
    def extra_repr(self) -> str:
        return f"in_feats {self.in_feats} out_feats {self.out_feats} first_stride {self.first_stride}"
 
# %%
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, inner_feats: int, out_feats: int, first_stride: int = 1):
        '''
            An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.
        '''
        super().__init__()

        # In the paper we can see that the first block never has first_strde > 1
        
        ## Can use either Sequential or ModuleList
        residual_blocks = []
        residual_blocks.append(ResidualBlock(in_feats=in_feats, 
                                             inner_feats=inner_feats,
                                                 out_feats=out_feats,
                                                 first_stride=first_stride))
        
        for _ in range(1, n_blocks):
            residual_blocks.append(ResidualBlock(in_feats=out_feats,
                                                 inner_feats=inner_feats,
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
class ResNet50(nn.Module):
    def __init__(
            self,
            n_blocks_per_group = (3, 4, 6, 3),
            in_features_per_group = (64, 256, 512, 1024),
            inner_features_pe_group = (64, 128, 256, 512),
            out_features_per_group = (256, 512, 1024, 2048),
            first_strides_per_group = (1, 2, 2, 2),
            n_classes = 1000
    ):
        super().__init__()
        self.n_blocks_per_group = n_blocks_per_group
        self.in_features_per_group = in_features_per_group
        self.inner_features_per_group = inner_features_pe_group
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
        
        # conv2, conv3, conv4, conv5, If i do this, i need to put in Sequential / nn.ModuleList
        # Otherwise, I could not use like blocks[i] as this won't pass parameters
        blocks = []
        for i in range(len(n_blocks_per_group)):
            blocks.append(BlockGroup(n_blocks = n_blocks_per_group[i],
                                          in_feats = in_features_per_group[i],
                                          inner_feats= inner_features_pe_group[i],
                                          out_feats = out_features_per_group[i],
                                          first_stride = first_strides_per_group[i]))


        # This blocks view of summary 
        self.blocks = nn.Sequential(*blocks)

        # average pool
        self.avg_pool = t.nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = t.nn.Flatten()
        self.linear = nn.Linear(in_features = 2048, out_features=n_classes)


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
            x: shape (batch, channels, height, width)
            Return: shape (batch, n_classes)
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2_maxpool(x)

        x = self.blocks(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x

# %%
# Veryfing your implementation by loading weights
def copy_weights(my_resnet : ResNet50, pretrained_resnet: torchvision.models.resnet34, copy=False) -> ResNet50:
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
    my_resnet = ResNet50()
    pretrained_resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    print(summary(my_resnet, input_data = torch.rand(2, 3, 32, 32), verbose=0, depth=10))
    print(summary(pretrained_resnet, input_data = torch.rand(2,3,32,32)))
#%%
## This is to load pre-trained weights
if MAIN and debug:
    my_resnet = ResNet50()
    pretrained_resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    my_resnet = copy_weights(my_resnet, pretrained_resnet)
# %%
## This is to display pandas frame between parameters and check
if MAIN and debug:
    my_resnet = ResNet50()
    pretrained_resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    print_param_count(my_resnet, pretrained_resnet)
# %%

def get_dataset(subset: int = 1):
    IMAGE_SIZE = 32
    MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomCrop(IMAGE_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean = MEAN, std=STD)
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = MEAN, std=STD)
    ])

    trainset = datasets.CIFAR100(root = './data', train=True, download=True,
                                      transform=TRAIN_TRANSFORM)
    testset = datasets.CIFAR100(root = './data', train=False, download=True,
                                     transform=TEST_TRANSFORM)
    print(f"Length of train {len(trainset)} of test: {len(testset)}") 
    if subset > 1:
        trainset = Subset(trainset, indices=range(0, len(trainset), subset))
        testset = Subset(testset, indices=range(0, len(testset), subset))
        print(f"Subset: Length of train {len(trainset)} of test: {len(testset)}") 

    return trainset, testset


@dataclass
class ResnetTrainingArgs():
    batch_size: int = 128 # 128
    epochs:int = 140 # 140
    optimizer: Type[t.optim.Optimizer] = t.optim.SGD
    learning_rate: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    n_classes: int = 100
    subset: int = 1 # 1
    wandb_project: Optional[str] = "resnet50"
    wandb_name: Optional[str] = "cifar100"


# %%
## Full tranining code
class ResnetTrainer():
    def __init__(self, args: ResnetTrainingArgs):
        self.args = args
        #self.model = ResNet50(n_classes=args.n_classes).to(device)
        self.model = torchvision.models.resnet50().to(device)
        self.optimizer = self.args.optimizer(self.model.parameters(),
                                            lr=args.learning_rate, 
                                            weight_decay=args.weight_decay,
                                            momentum=args.momentum)
        self.trainset, self.testset = get_dataset(args.subset)

        # Store per epoch, not per timestep
        self.step = 0 # gloabl step = epochs * len(data_loader)
        wandb.init(
            project = args.wandb_project,
            name = args.wandb_name,
            config = self.args
        )

        wandb.save('resnet50.py')

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.args.batch_size,
                          shuffle=True, pin_memory=False)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.args.batch_size,
                          shuffle=False, pin_memory=False)
    
    def train_one_epoch(self, epoch_index: int):
        train_loss = 0 # Log at end of epoch
        train_acc = 0 # Log at end of epooch
        running_loss = 0 # Running loss at each forward/backward pass per epoch
        running_acc = 0 # Running accuracy
        start = time.time()
        
        train_dataloader = self.train_dataloader()

        # Must set to train
        self.model.train()

        for i, data in enumerate(train_dataloader):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            self.optimizer.zero_grad()

            # Do predictions
            outputs = self.model(imgs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            # Do optimization
            self.optimizer.step()

            # Gathering data and report
            # Loss of batch by multiplying averaged loss by sample size
            train_loss += loss.item() * imgs.size(0)

            # Accuracy 
            preds = torch.argmax(outputs, dim=-1)
            eq = (preds == labels).float().mean()
            train_acc += eq.item() * imgs.size(0)

            ## Running loss /acc
            running_loss += loss.item()
            running_acc += eq.item()

            # It only makes sense to print per-step loss/accuracy
            # instead of train_loss and train_acc which is not **averaged yet**
            if (i+1) % 10 == 0:
                print(f"Train: Epoch {epoch_index}|{self.args.epochs}." \
                        f" Done {100*(i+1)/len(train_dataloader):.2f}. " \
                        f" Time {(time.time() - start):.2f} elapsed"\
                        f" Running Loss: {running_loss/(i+1):.2f} Running Acc {running_acc/(i+1):.2f} ")

            self.step+=1
            wandb.log({"train_running_loss": running_loss/(i+1), "train_running_acc" : running_acc/(i+1)}, step=self.step)
                
        # At the end store to logged
        train_loss = train_loss / len(train_dataloader.dataset)
        train_acc = train_acc / len(train_dataloader.dataset)

        wandb.log({"train_loss": train_loss, "train_acc" : train_acc}, step=self.step)

        return train_loss, train_acc

    def validation_one_epoch(self, epoch_index):
        val_loss = 0
        val_acc = 0
        running_loss = 0
        running_acc = 0
        start = time.time()
        test_dataloader = self.test_dataloader()

        self.model.eval()

        with torch.no_grad():

            for i, data in enumerate(test_dataloader):
                imgs, labels = data
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = self.model(imgs)
                loss = F.cross_entropy(outputs, labels)

                # Validation loss
                val_loss += loss.item() * imgs.size(0)


                # Validation accuracy
                preds = torch.argmax(outputs, dim=-1)
                eq = (preds == labels).float().mean()
                val_acc += eq.item() * imgs.size(0)

                ## Running loss /acc
                running_loss += loss.item()
                running_acc += eq.item()

                if (i+1) % 5 == 0:
                    print(f"Val: Epoch {epoch_index}|{self.args.epochs}." \
                        f"Done {100*(i+1)/len(test_dataloader):.2f}. " \
                        f"Time {(time.time() - start):.2f} elapsed"\
                        f" Running Loss: {running_loss/(i+1):.2f} Running Acc {running_acc/(i+1):.2f} ")

        # At the end store to logged
        val_loss = val_loss / len(test_dataloader.dataset)
        val_acc = val_acc / len(test_dataloader.dataset)

        wandb.log({"val_loss": val_loss, "val_acc" : val_acc}, step=self.step)

        return val_loss, val_acc

    def train(self):
        for epoch in range(self.args.epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)

            val_loss, val_acc = self.validation_one_epoch(epoch)

            ## At this stage, you are now able to print loss/acc of whole dataset
            # instead of per-step like in train_one_epoch() function

            print(f"Epoch {epoch} | {self.args.epochs}."\
                  f" Train loss: {train_loss:.2f} Train acc: {train_acc:.2f}" \
                 f" Val loss: {val_loss:.2f} Val acc: {val_acc:.2f}")

            # Save model if val_loss < val_loss_min


args = ResnetTrainingArgs()
trainer = ResnetTrainer(args)
trainer.train()
# %%
