import torch
import os
import json
from torchvision import models
import torch.nn as nn


MODEL_NAME = 'model_resnet18.pth'
JSON_CONTENT_TYPE = 'application/json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model from the model path
def model_fn(model_dir):
    print("Loading model.")
    model_path = '{}/{}'.format(model_dir, MODEL_NAME)
    print("Loading model.")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("Model loaded.")
    model.to(DEVICE)
    print("Moved to device")
    return model


# Preprocess the input data
def input_fn(request_body, request_content_type):
    print(f"Deserializing the input data with content type: {request_content_type}")
    if request_content_type == JSON_CONTENT_TYPE:
        data = json.loads(request_body)
        input_data = torch.tensor(data['inputs'], dtype=torch.float32)
        input_data = input_data.to(DEVICE)
        print("Data loaded")
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


# Postprocess the output data
def output_fn(prediction, response_content_type):
    print(f"Serializing the output data with content type: {response_content_type}")
    if response_content_type == JSON_CONTENT_TYPE:
        result = prediction.detach().cpu().numpy().tolist()
        return json.dumps({'outputs': result})
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")


# Perform inference on the input data
def predict_fn(input_data, model):
    print("Performing prediction.")
    model.eval()
    print("Model to eval - done")
    with torch.no_grad():
        prediction = model(input_data)
    print("Predicted")
    pred = prediction.argmax(dim=1, keepdim=True)[0][0]
    print("Get Pred")
    return pred



