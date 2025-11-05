import mlflow.pytorch
import torch.nn.functional as F

model_name = "Best_CNN"
model_version = 1

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pytorch.load_model(model_uri)


model = mlflow.pytorch.load_model(model_uri)
example_input = torch.randn(1, 3, 32, 32)

model.eval()
with torch.no_grad():
    output = model(example_input)

print(f"Raw: {output}")
probs = F.softmax(output, dim =1 )

print(f"Probabilities {probs*100}")

predicted = torch.argmax(probs, dim = 1)

print(f"Prediction: {predicted.item()}" )









import requests
import torch
import json
import numpy as np

# API URL des Modell-Servers
url = "http://127.0.0.1:5007/invocations"

example_input = torch.randn(1, 3, 32, 32).numpy().tolist()  # Modell erwartet NumPy-Array

input_data = json.dumps({"instances": example_input})

response = requests.post(url, data=input_data, headers={"Content-Type": "application/json"})

print("Modell-Inferenz Ergebnis:", F.softmax(torch.tensor(response.json()["predictions"][0]), dim = 0))
