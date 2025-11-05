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
