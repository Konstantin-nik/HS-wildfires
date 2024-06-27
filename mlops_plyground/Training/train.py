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

import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision import models
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sagemaker.feature_store.feature_group import FeatureGroup

def download_images(metadata, download_dir='/opt/ml/input/data/images'):
    s3_client = boto3.client('s3')
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    for record in metadata:
        image_location = record['image_location']
        bucket, key = image_location.replace('s3://', '').split('/', 1)
        local_path = os.path.join(download_dir, os.path.basename(key))
        
        s3_client.download_file(bucket, key, local_path)

        record['local_path'] = local_path  # Add the local path to the record

    return metadata

class FireDataset(Dataset):
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata[idx]['local_path']
        image = Image.open(img_path).convert('RGB')
        label = self.metadata[idx]['label']
        
        if self.transform:
            image = self.transform(image)

        return image, label

def train(model, train_loader, optimizer, loss_function, epoch, device):
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

def test(model, test_loader, loss_function, epoch, device):
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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train_test(model, optimizer, train_loader, test_loader, device, n_epochs=1):
    for epoch in range(0, n_epochs):
        train(model, train_loader, optimizer, criterion, epoch, device)
        test(model, test_loader, criterion, epoch, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    
    args = parser.parse_args()

    boto_session = boto3.Session()
    region = boto_session.region_name

    sm_session = sagemaker.Session()
    sm_client = boto_session.client("sagemaker")
    sm_role = sagemaker.get_execution_role()

    feature_group_name = 'fire-image-feature-group'

    athena_client = boto3.client('athena', region_name=region)

    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sm_session)

    query = """SELECT *
    FROM "AwsDataCatalog"."sagemaker_featurestore"."fire_image_feature_group_1718694943";
    """

    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': 'sagemaker_featurestore'
        },
        ResultConfiguration={
            'OutputLocation': 's3://wildfires/feature-store-output/'
        }
    )

    query_execution_id = response['QueryExecutionId']

    status = 'RUNNING'
    while status != 'SUCCEEDED':
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        status = response['QueryExecution']['Status']['State']

    response = athena_client.get_query_results(QueryExecutionId=query_execution_id)

    rows = [row['Data'] for row in response['ResultSet']['Rows'][1:]]
    columns = [col['VarCharValue'] for col in response['ResultSet']['Rows'][0]['Data']]

    metadata = [
        {
            'image_id': row[0]['VarCharValue'],
            'image_location': row[1]['VarCharValue'],
            'label': int(row[2]['VarCharValue']),
            'image_type': row[3]['VarCharValue'],
            'event_time': row[4]['VarCharValue'],
        } for row in rows
    ]

    metadata = download_images(metadata)

    train_metadata, test_metadata = train_test_split(metadata, test_size=0.2, stratify=[m['label'] for m in metadata], random_state=42)
    train_metadata, val_metadata = train_test_split(train_metadata, test_size=0.25, stratify=[m['label'] for m in train_metadata], random_state=42)

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

    train_dataset = FireDataset(train_metadata, transform=train_transform)
    val_dataset = FireDataset(val_metadata, transform=val_test_transform)
    test_dataset = FireDataset(test_metadata, transform=val_test_transform)

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

    train_test(model=model,
               optimizer=optimizer,
               train_loader=train_loader,
               test_loader=test_loader,
               device=device,
               n_epochs=args.num_epochs
               )
