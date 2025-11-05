import mlflow.pytorch
import torch.nn.functional as F

model_name = "Best_CNN"
model_version = 1

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pytorch.load_model(model_uri)
