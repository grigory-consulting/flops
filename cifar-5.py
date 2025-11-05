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







# data loader 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



with mlflow.start_run() as run:
        mlflow.log_param("optimizer", "adam") # 
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("conv1", conv1)
        mlflow.log_param("conv2", conv2)
        mlflow.log_param("fc1", fc1)

        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                optimizer.zero_grad() # Gradienten auf Null setzen
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Gradient descent
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0) 
            
            accuracy = correct / total 
            mlflow.log_metric("loss", running_loss, step = epoch)
            mlflow.log_metric("train_accuracy", accuracy, step = epoch)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Train accuracy: {accuracy:.4f}")
        
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                predicted = torch.max(outputs.data, 1)[-1] # argmax von outputs
                total += labels.size(0) # Gesamte Anzahl der Labels
                correct += (predicted==labels).sum().item() # Anzahl der richtig klassifizierten Labels
        
        accuracy = correct / total
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.pytorch.log_model(model, "model" , input_example=input_example)
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)







