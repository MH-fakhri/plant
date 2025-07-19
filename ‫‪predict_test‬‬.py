import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
num_classes = 28  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = models.efficientnet_b3(weights=None)  
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)


model.load_state_dict(torch.load('/content/drive/MyDrive/B3_1_model_weights.pth', map_location=device))
model.eval()
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder('/content/drive/MyDrive/PlantDoc-Dataset-master/test', transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.classes


while len(class_names) < num_classes:
    class_names.append(f"class_{len(class_names)}")

y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


acc = accuracy_score(y_true, y_pred)
print(f"\n Accuracy: {acc:.4f}")

cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

report = classification_report(
    y_true, y_pred,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0
)
print("\n📋 Classification Report:")
print(report)


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img_tensor = img_tensor.cpu() * std + mean
    return img_tensor.permute(1, 2, 0).numpy()


sample_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
n = 6  
shown = 0

plt.figure(figsize=(15, 10))
with torch.no_grad():
    for inputs, labels in sample_loader:
        if shown >= n:
            break
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        is_correct = preds.item() == labels.item()
        color = 'green' if is_correct else 'red'

        plt.subplot(2, 3, shown + 1)
        img = denormalize(inputs[0])
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Pred: {class_names[preds.item()]}\nTrue: {class_names[labels.item()]}", color=color)
        shown += 1

plt.tight_layout()
plt.show()