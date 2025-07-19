import torch
from tqdm import tqdm
from torcheval.metrics import MulticlassAccuracy, Mean
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
device = 'cuda' if torch.cuda.is_available() else "cpu"
print(device)
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

def compute_class_weights(loader):
    class_counts = np.zeros(len(train_set.classes))
    total_count = 0
    for _, labels in loader:
        for label in labels:
            class_counts[label.item()] += 1
            total_count += 1
    class_weights = total_count / (len(train_set.classes) * class_counts)
    return torch.tensor(class_weights, dtype=torch.float).to(device)
train_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(degrees=20),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.ToTensor(),
    # AddGaussianNoise(),

    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
valid_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_set = ImageFolder("/content/drive/MyDrive/PlantDoc-Dataset-master/train", transform=train_transforms)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=6)

valid_set = ImageFolder("/content/drive/MyDrive/PlantDoc-Dataset-master/test", transform=valid_transform)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=True, num_workers=6)
weight_s = compute_class_weights(loader=train_loader)


def train_loop(model, loader, loss_fn, optimizer, device):

    model.train()
    mean_loss = Mean().to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=28, device=device)

    with tqdm(loader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            tepoch.set_description(f"Training Epoch")

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            mean_loss.update(loss)
            accuracy_metric.update(outputs, labels)

            tepoch.set_postfix(loss=mean_loss.compute().item(), acc=accuracy_metric.compute().item())

    avg_loss = mean_loss.compute().item()
    accuracy = accuracy_metric.compute().item()
    return avg_loss, accuracy

def validation_loop(model, loader, loss_fn, device):

    model.eval()
    mean_loss = Mean().to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=28, device=device)

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            mean_loss.update(loss)
            accuracy_metric.update(outputs, labels)

    avg_loss =  mean_loss.compute().item()
    accuracy = accuracy_metric.compute().item()
    return avg_loss, accuracy
model_B3 = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)


num_classes = len(train_set.classes)
model_B3.classifier = nn.Sequential(
    nn.Dropout(p=0.5), 
    nn.Linear(model_B3.classifier[1].in_features, num_classes)
)
model_B3 = model_B3.to(device)
optimizer = optim.AdamW(model_B3.parameters(), lr=1e-4, weight_decay=1e-4) 
loss_fn = nn.CrossEntropyLoss() #without weigts
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.1)  
for i in range(16):
    train_loop(model_B3, train_loader, loss_fn, optimizer, device)
    print(validation_loop(model_B3, valid_loader, loss_fn, device))
torch.save(model_B3.state_dict(), 'newB3model_weights.pth')
