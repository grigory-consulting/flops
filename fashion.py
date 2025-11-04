import torch
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets, transforms
import pandas as pd


transform = transforms.Compose([
    transforms.ToTensor(),
])



train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)






model.eval() 
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = torch.max(outputs.data, 1)[-1] 
            total += labels.size(0) 
            correct += (predicted==labels).sum().item()

    accuracy = correct/total
    print(f"Accuracy: {accuracy*100:.2f}%")
    mlflow.log_metric("accuracy", accuracy)
