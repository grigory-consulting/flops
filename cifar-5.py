import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
import onnx # 
import time
from mlflow.types import Schema, TensorSpec
import numpy as np
import optuna #
import mlflow



transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


selected_classes = [0, 1, 2, 3, 4]


train_indices = [i for i, label in enumerate(train_dataset.targets) if label in selected_classes]
test_indices = [i for i, label in enumerate(test_dataset.targets) if label in selected_classes]


train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
test_dataset = torch.utils.data.Subset(test_dataset, test_indices)


input_example = torch.randn(1,3,32,32).numpy()


input_schema = Schema([TensorSpec(np.dtype(np.float32),(-1, 3, 32, 32)) # beliebige Größe des batch size's
])
output_schema = Schema([
    TensorSpec(np.dtype(np.float32), (-1, 10))
])
signature = mlflow.models.signature.ModelSignature(inputs=input_schema, outputs=output_schema)
