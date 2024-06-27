import time
import os
import json
import boto3
import numpy as np
import sagemaker
import requests
import torch
import tqdm
import argparse
import pickle
from pathlib import Path
import mlflow

import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision import models
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

tracking_server_arn = os.environ.get('MLFLOW_TRACKING_URIs', None)
mlflow_experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', None)

model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
model_path = os.path.join(model_dir, 'model.pth')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--run-name', type=str, default="")
    parser.add_argument('--learning-rate', type=float, default=0.1)

    args = parser.parse_args()

    return args

class FireDataset(Dataset):
    def __init__(self, metadata, train_dir="", transform=None):
        self.metadata = metadata
        self.transform = transform
        self.train_dir = train_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # img_path = self.metadata[idx]['local_path']
        img_path = os.path.basename(self.metadata[idx]['image_location'])
        img_path = os.path.join(self.train_dir, img_path)
        image = Image.open(img_path).convert('RGB')
        label = self.metadata[idx]['label']
        
        if self.transform:
            image = self.transform(image)

        return image, label

def train(model, train_loader, optimizer, loss_function, epoch, device, run):
    model = model.to(device)
    loss_function = loss_function.to(device)
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        train_loss += loss.sum().item()
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)
    
    mlflow.log_metric('training_loss', train_loss, step=epoch, run_id=run.info.run_id)

def test(model, test_loader, loss_function, epoch, device, run):
    model = model.to(device)
    loss_function = loss_function.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    mlflow.log_metric('test_loss', test_loss, step=epoch, run_id=run.info.run_id)
    mlflow.log_metric('test_acc', correct / len(test_loader.dataset), step=epoch, run_id=run.info.run_id)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train_test(model, optimizer, train_loader, test_loader, device, run, n_epochs=1):
    for epoch in range(0, n_epochs):
        train(model, train_loader, optimizer, criterion, epoch, device, run)
        test(model, test_loader, criterion, epoch, device, run)


if __name__ == '__main__':
    args = parse_args()

    boto_session = boto3.Session()
    region = boto_session.region_name

    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(mlflow_experiment_name)

    print('####', tracking_server_arn, mlflow_experiment_name)

    # ----------------------------------------------

    with open(os.path.join(args.train_dir, 'train.pkl'), 'rb') as f:
        train_metadata = pickle.load(f)

    with open(os.path.join(args.train_dir, 'val.pkl'), 'rb') as f:
        val_metadata = pickle.load(f)

    with open(os.path.join(args.train_dir, 'test.pkl'), 'rb') as f:
        test_metadata = pickle.load(f)

    print ('## Loaded file')
    # -----------------------------------------------

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FireDataset(train_metadata, train_dir=args.train_dir, transform=train_transform)
    val_dataset = FireDataset(val_metadata, train_dir=args.train_dir, transform=val_test_transform)
    test_dataset = FireDataset(test_metadata, train_dir=args.train_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    with mlflow.start_run(run_name=sagemaker.utils.name_from_base(args.run_name)) as run:
        mlflow.autolog()

        mlflow.log_params({
            'epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'train size': len(train_metadata),
            'validation size': len(val_metadata),
            'test size': len(test_metadata),
            'learning rate': args.learning_rate,
        }, run_id=run.info.run_id)

        train_test(model=model,
                   optimizer=optimizer,
                   train_loader=train_loader,
                   test_loader=test_loader,
                   device=device,
                   n_epochs=args.num_epochs,
                   run=run,
                   )

        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, artifact_path="models")

        mlflow.end_run(status='FINISHED')
