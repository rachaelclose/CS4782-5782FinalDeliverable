# Data Used

We used PyTorch's built-in CIFAR10 dataset. We normalized, split, and loaded the dataset according to the paper, as described in the code below:

```
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
```