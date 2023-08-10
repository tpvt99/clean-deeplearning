#%%
import torch
import torchvision
from torch import nn
from jaxtyping import Float
from torchinfo import summary
import wandb
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import time
from typing import Optional

debug = False # for shell debugging

MAIN = __name__ == "__main__"
# %%
class LocalResponseNormalization(nn.Module):
    def __init__(self, k: int, n: int, alpha: float, beta: float):
        pass

    def forward(x: Float[torch.Tensor, "batch c h w"]) -> Float[torch.Tensor, "batch c h w"]:
        # pad channel
        x_pad = torch.nn.functional.pad(x, pad = (n/2, n/2, 0, 0, 0 ,0), mode="constant", value=0)



class Alexnet(nn.Module):
    def __init__(self, dropout:float = 0.5, num_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            # Inputs (3, 32, 32), Outputs (64, 8, 8)
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5, stride= 4, padding =2,
                      bias=True),
            nn.ReLU(),
            nn.LocalResponseNorm(k=2, size =5, alpha=1e-4, beta=0.75),
            # Inputs (64, 8, 8), Outputs (64, 4, 4)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Inputs (64, 4, 4), Outputs (192, 4, 4)
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.LocalResponseNorm(k=2, size =5, alpha=1e-4, beta=0.75),
            # Inputs (192, 4, 4), Outputs (192, 2, 2)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Inputs (192, 2, 2), Outputs (384, 4, 4)
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Inputs (384, 13, 13), Outputs (384, 13, 13)
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Inputs (384, 13, 13), Outputs (256, 13, 13)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Inputs (256, 2, 2), Outputs (256, 1, 1)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten()

        self.classifcation = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256*1*1, 512),
            nn.ReLU(),

            nn.Dropout(p = dropout),
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, num_classes)
            
        )

    def forward(self, x : Float[torch.Tensor, "batch c h w"]) -> Float[torch.Tensor, "batch n"]:
        x = self.features(x)

        x = self.avgpool(x)
        x = self.flatten(x)

        x = self.classifcation(x)

        return x

# %%
if MAIN and debug:
    my_alexnet = Alexnet()
    pretrained_alexnet = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    print(summary(my_alexnet, input_data = torch.rand(2, 3, 32, 32), verbose=0, depth=10))
    print(summary(pretrained_alexnet, input_data = torch.rand(2,3,32,32)))
# %%
## Loading and printing. If not error than your implemented model is correct
from utils import print_param_count
if MAIN and debug:
    my_alexnet = Alexnet()
    pretrained_alexnet = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    print_param_count(my_alexnet, pretrained_alexnet)
#
# %%
@dataclass
class AlexNetTrainingArgs:
    image_size: int = 32
    batch_size: int = 128
    epochs: int = 600
    loss_label_smoothing = 0.1

    model_lr = 0.1
    model_momentum = 0.9
    model_weight_decay = 2e-5
    model_ema_decay = 0.99998
    dropout = 0.5
    num_classes = 100

    lr_scheduler_t0 = epochs//4
    lr_scheduler_t_mult = 1
    lr_scheduler_eta_min = 5e-5

    train_print_frequency = 20
    valid_print_frequency = 20

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device('cpu')

    wandb_project: Optional[str] = "alexnet"
    wandb_name: Optional[str] = "cifar100"

def get_imagenet(args: AlexNetTrainingArgs):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size),
        transforms.TrivialAugmentWide(),
        transforms.RandomRotation([0, 270]),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop([args.image_size, args.image_size]),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    #test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset

def get_cifar100(args: AlexNetTrainingArgs):
    train_transform = transforms.Compose([
        #transforms.RandomResizedCrop(args.image_size),
        #transforms.TrivialAugmentWide(),
        #transforms.RandomRotation([0, 270]),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),

        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop([args.image_size, args.image_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset

class AlexnetTrainer():
    def __init__(self, args: AlexNetTrainingArgs):
        self.args = args
        self.model = Alexnet(dropout = args.dropout, num_classes=args.num_classes).to(self.args.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                      lr = args.model_lr,
                                      momentum = args.model_momentum,
                                      weight_decay = args.model_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                              T_0 = args.lr_scheduler_t0,
                                              T_mult = args.lr_scheduler_t_mult,
                                              eta_min = args.lr_scheduler_eta_min)
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.loss_label_smoothing)
        self.trainset, self.testset = get_cifar100(args)

        self.step = 0

        wandb.init(
            project = args.wandb_project,
            name = args.wandb_name,
            config = self.args
        )
        wandb.save('alexnet.py')

    def get_train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.args.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)
    
    def get_test_dataloader(self):
        return DataLoader(self.testset,
                          batch_size=self.args.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)

    def train_one_epoch(self, epoch_index, train_dataloader):
        num_batches = len(train_dataloader)
        start = time.time()

        # My log
        train_loss = 0
        train_acc = 0
        running_loss = 0
        running_acc = 0

        # Put model in train
        self.model.train()

        for index, data in enumerate(train_dataloader):
            imgs, labels = data
            imgs = imgs.to(self.args.device)
            labels = labels.to(self.args.device)

            self.optimizer.zero_grad()

            # Do prediction
            output = self.model(imgs)
            loss = self.criterion(output, labels)

            # backprop
            loss.backward()
            self.optimizer.step()

            ## My record
            acc = (output.argmax(dim=-1) == labels).float().mean()
            running_loss += loss.item()
            running_acc += acc.item()
            train_loss += loss.item() * imgs.size(0)
            train_acc += acc.item() * imgs.size(0)

            if self.step % self.args.train_print_frequency == 0:
                print(f"Train: Epoch {epoch_index}|{self.args.epochs}." \
                        f" Done {100*(index+1)/len(train_dataloader):.2f}%. " \
                        f" Time {(time.time() - start):.2f} elapsed"\
                        f" Running Loss: {running_loss/(index+1):.2f} Running Acc {running_acc/(index+1):.2f} ")

            self.step += 1
            wandb.log({"running_loss" : running_loss/(index+1), "running_acc": running_acc/(index+1)},
                      step = self.step)

        train_loss = train_loss / len(train_dataloader.dataset)
        train_acc = train_acc / len(train_dataloader.dataset)

        wandb.log({"train_loss": train_loss, "train_acc" : train_acc}, step=self.step)

        return train_loss, train_acc

    def validation_one_step(self, epoch_index, test_dataloader):
        start = time.time()

        # My log
        running_loss = 0
        running_acc = 0
        val_loss = 0
        val_acc = 0

        # Put model to validation mode
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                imgs, labels = data
                imgs = imgs.to(self.args.device)
                labels = labels.to(self.args.device)

                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                # Validation loss
                val_loss += loss.item() * imgs.size(0)

                # Validation accuracy
                acc = (outputs.argmax(dim=-1) == labels).float().mean()
                val_acc += acc.item() * imgs.size(0)

                ## Running loss /acc
                running_loss += loss.item()
                running_acc += acc.item()

                if (i+1) % self.args.valid_print_frequency == 0:
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
        train_dataloader = self.get_train_dataloader()
        test_dataloader = self.get_test_dataloader()

        for epoch in range(self.args.epochs):
            train_loss, train_acc = self.train_one_epoch(epoch, train_dataloader)

            val_loss, val_acc = self.validation_one_step(epoch, test_dataloader)

            # update scheduler
            #self.scheduler.step()

if MAIN:
    args = AlexNetTrainingArgs()
    alexnetTrainer = AlexnetTrainer(args)
    alexnetTrainer.train()
# %%
loss = nn.CrossEntropyLoss(label_smoothing=0.1)
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
print(output.item())
# %%
input.size(0)
# %%
