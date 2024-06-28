import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker", "boto3"])

import os
import boto3
import sagemaker
import torch
import argparse
import pickle
import numpy as np
import json
import tarfile

import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision import models
from torchvision import transforms
from botocore.exceptions import ClientError
from torch.utils.data import Dataset, DataLoader
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.pytorch import PyTorchModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--result_prefix', type=str, required=True)
    parser.add_argument('--result_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--region', type=str, required=True)

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


def test(model, test_loader, loss_function, device):
    model = model.to(device)
    loss_function = loss_function.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        # This try-except block could be removed. It is just for debugging
        try:
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                test_loss += loss_function(output, target).sum().item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        except FileNotFoundError as e:
            print(f"SageMaker FileNotFoundError occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            print("test_loader execution finished.")

    test_loss /= len(test_loader.dataset)
    acc = round(100. * correct / len(test_loader.dataset), 2)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, acc


def get_fs_last_vesrion(s3, bucket_name):
    version_file_path = "feature-store/fs_version.txt"
    local_file_path = "/tmp/fs_version.txt"
    try:
        s3.download_file(bucket_name, version_file_path, local_file_path)
        print(f"Version file successfully loaded: {local_file_path}")
        with open(local_file_path) as file:
            version = file.readline().strip()
        return int(version)
    except ClientError:
        print("No version file founded")
        return 0


def get_responce_athena(query, athena_client):
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': 'sagemaker_featurestore'  # Replace with your Athena database name
        },
        ResultConfiguration={
            'OutputLocation': 's3://wildfires/feature-store-output/'  # Replace with your S3 bucket
        }
    )

    query_execution_id = response['QueryExecutionId']

    status = 'RUNNING'
    while status != 'SUCCEEDED':
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        status = response['QueryExecution']['Status']['State']

    response = athena_client.get_query_results(QueryExecutionId=query_execution_id)

    return response


def get_metadata(response):
    rows = [row['Data'] for row in response['ResultSet']['Rows'][1:]]

    metadata = [
        {
            'image_id': row[0]['VarCharValue'],
            'image_location': row[1]['VarCharValue'].split('/')[-1],
            'label': int(row[2]['VarCharValue']),
            'image_type': row[3]['VarCharValue'],
            'event_time': row[4]['VarCharValue'],
        } for row in rows
    ]

    return metadata


if __name__ == '__main__':
    args = parse_args()

    s3 = boto3.client('s3')
    athena_client = boto3.client('athena', region_name=args.region)
    sm_session = sagemaker.Session(boto3.Session(region_name=args.region))
    sm_role = sagemaker.get_execution_role(sm_session)
    sagemaker_client = sm_session.boto_session.client('sagemaker')

    print('Evaluation job is started')

    print('Getting Metadata')
    feature_group_name = f'wildfire-feature-group-v{get_fs_last_vesrion(s3, args.bucket)}'

    feature_group = FeatureGroup(
        name=feature_group_name, sagemaker_session=sm_session
    )

    fs_train_query = feature_group.athena_query()

    query = f"""
    SELECT *
    FROM "{fs_train_query.table_name}"
    WHERE data_purpose = 'test';
    """

    response = get_responce_athena(query, athena_client)
    test_metadata = get_metadata(response)

    print('## Metadata Loaded')
    print(f'{len(test_metadata)} Metadata loaded')
    # -------------------------------------------

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = FireDataset(test_metadata, train_dir=args.data_path, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    s3.download_file('wildfires', 'code/inference.py', 'code/inference.py')
    model.load_state_dict(torch.load(f'{args.model_path}/model.pth', map_location=device))
    print('Best Model is loaded')

    criterion = nn.CrossEntropyLoss()

    print('Testing will start')

    test_loss, acc = test(
        model=model,
        test_loader=test_loader,
        loss_function=criterion,
        device=device
    )

    print('Testing is ended')

    result = {
        'metrics': {
            'accuracy': acc,
            'test_loss': test_loss
        }
    }
    print('Result is calculated')
    print(result)

    # result_file = args.result_file + model_name
    # s3.put_object(Bucket=args.result_bucket, Key=f'{args.result_prefix}/{result_file}', Body=json.dumps(result))

    os.makedirs('./code', exist_ok=True)
    s3.download_file('wildfires', 'code/inference.py', 'code/inference.py')

    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        tar.add('model.pth', arcname=os.path.basename('model.pth'))
        tar.add('code', arcname=os.path.basename('code'))

    s3.put_object(Bucket=args.bucket, Key='models/last/model.tar.gz', Body='model.tar.gz')

    model = PyTorchModel(
        model_data='model.tar.gz',
        role=sm_role,
        framework_version='2.0.0',
        py_version='py310',
        entry_point='inference.py'
    )

    # Model Package
    model_package = model.register(
        content_types=['application/json'],    # Input type JSON
        response_types=['application/json'],   # Output type JSON
        inference_instances=['ml.g5.xlarge'],  # Inference instance type
        transform_instances=['ml.g5.xlarge'],  # Transform instance type
        model_package_group_name='wildfire-model-group-name'  # Model Group Name
    )

    with open('/opt/ml/processing/output/model_arn.json', mode='w') as f:
        json.dumps({"ARN": model_package.model_package_arn}, f)

    sm_session.update_model_package(
        ModelPackageArn=model_package.model_package_arn,
        CustomerMetadataProperties={"acc": f"{acc}", "test_loss": f"{test_loss}"}
    )
    print('Results are registered to Model Registry')

    print('Job is Finished')
