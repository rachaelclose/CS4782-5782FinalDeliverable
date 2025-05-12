import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.ops import stochastic_depth
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import ResNetDrop
import ResNet

#Cifar-10 Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

full_train_aug = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True,
    transform=transform_train
)
full_train_det = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False,
    transform=transform_test
)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False,
    transform=transform_test
)

num_train = len(full_train_aug)
val_size  = 5000
train_size = num_train - val_size
indices = torch.randperm(num_train).tolist()
train_idx, val_idx = indices[:train_size], indices[train_size:]

train_subset = Subset(full_train_aug, train_idx)
val_subset   = Subset(full_train_det, val_idx)

batch_size = 128
trainloader = DataLoader(train_subset, batch_size=batch_size,
                         shuffle=True,  num_workers=2, pin_memory=True)
valloader   = DataLoader(val_subset,   batch_size=batch_size,
                         shuffle=False, num_workers=2, pin_memory=True)
testloader  = DataLoader(testset,      batch_size=batch_size,
                         shuffle=False, num_workers=2, pin_memory=True)

base_dir = "./checkpoints"
os.makedirs(base_dir, exist_ok=True)

best_val_loss = float('inf')

def validate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    correct  = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss   = criterion(output, target)
            val_loss += loss.item() * data.size(0)
            preds    = output.argmax(dim=1)
            correct += (preds == target).sum().item()
    val_loss /= len(loader.dataset)
    val_acc   = 100. * correct / len(loader.dataset)
    return val_loss, val_acc

def train(model, optimizer, scheduler, criterion, epochs, save_name, history=None):
    """
    epochs: number of epochs to run
    save_name: filename under base_dir/ where best model is saved
    """
    global best_val_loss
    best_val_loss = float('inf')

    if history is None:
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': []
        }

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss   = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            if batch_idx % 100 == 0:
                pct = 100. * batch_idx / len(trainloader)
                print(f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(trainloader.dataset)}"
                      f" ({pct:.0f}%)]\tLoss: {loss.item():.6f}")

        avg_train = train_loss / len(trainloader.dataset)
        print(f"====> Epoch {epoch} Average train loss: {avg_train:.4f}")

        val_loss, val_acc = validate(model, valloader, criterion)
        print(f"====> Validation loss: {val_loss:.4f},  acc: {val_acc:.2f}%")

        test_loss, test_acc = test(model, criterion)
        print(f"====> Test loss: {test_loss:.4f},  acc: {test_acc:.2f}%")

        history['train_loss'].append(avg_train)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        scheduler.step()

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(base_dir, save_name))
            print(f"====> New best model saved to {save_name}\n")
        else:
            print()

    # return model
    return history

def test(model, criterion, ckpt_name=None):
    """
    Runs on testloader. If ckpt_name is given, loads it first.
    """
    if ckpt_name is not None:
        model.load_state_dict(torch.load(os.path.join(base_dir, ckpt_name)))
    model.to(device).eval()

    test_loss = 0.0
    correct   = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss   = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            preds     = output.argmax(dim=1)
            correct  += (preds == target).sum().item()

    test_loss /= len(testloader.dataset)
    test_acc   = 100. * correct / len(testloader.dataset)
    print(f"====> Test set loss: {test_loss:.4f},  acc: {test_acc:.2f}%")
    return test_loss, test_acc
  
num_blocks = 18
base_resnet = ResNet(num_blocks).to(device)
drop_resnet = ResNetDrop(num_blocks).to(device)

criterion = torch.nn.CrossEntropyLoss()

################# CHANGE EPOCHS HERE IF YOU WANT TO TEST #######################
# Almost make sure to change scheduler_b and scheduler_d accordingly
epochs = 500
# epochs = 5
################################################################################

# Baseline ResNet
optimizer_b = optim.SGD(
    base_resnet.parameters(),
    lr=0.1, momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

####### CHANGE SCHEDULER_B & D TO BE MILESTONES 1/2 and 3/4 OF EPOCHS ########
scheduler_b = torch.optim.lr_scheduler.MultiStepLR(
    optimizer_b, milestones=[250, 375], gamma=0.1)

print(">>> Training baseline ResNet")
hist_base = train(base_resnet, optimizer_b, scheduler_b, criterion, epochs, save_name="best_base.pth")

# Stochastic‐depth ResNetDrop
optimizer_d = optim.SGD(
    drop_resnet.parameters(),
    lr=0.1, momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
    optimizer_d, milestones=[250, 375], gamma=0.1)
# scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer_d, milestones=[2, 4], gamma=0.1)
################################################################################

print("\n>>> Training stochastic-depth ResNetDrop")
hist_drop = train(drop_resnet, optimizer_d, scheduler_d, criterion, epochs, save_name="best_drop.pth")
