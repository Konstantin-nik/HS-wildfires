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
from torch.utils.data import Dataset, DataLoader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_metadata_bucket', type=str, required=True)
    parser.add_argument('--test_metadata_prefix', type=str, required=True)
    parser.add_argument('--test_metadata_file', type=str, required=True)
    parser.add_argument('--best_model_bucket', type=str, required=True)
    parser.add_argument('--best_model_prefix', type=str, required=True)
    parser.add_argument('--best_model_file', type=str, required=True)
    parser.add_argument('--result_bucket', type=str, required=True)
    parser.add_argument('--result_prefix', type=str, required=True)
    parser.add_argument('--result_file', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_package_arn', type=str, required=True)
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


if __name__ == '__main__':
    args = parse_args()

    boto_session = boto3.Session()
    region = boto_session.region_name

    print(region)
    print(args.region)

    sm = boto3.client('sagemaker', region_name='eu-central-1')

    s3 = boto3.client('s3')
    print('Evaulation job is started')

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_metadata_bucket = args.test_metadata_bucket
    test_metadata_prefix = args.test_metadata_prefix
    test_metadata_file = args.test_metadata_file

    s3.download_file(test_metadata_bucket, f'{test_metadata_prefix}/{test_metadata_file}', test_metadata_file)
    with open(test_metadata_file, 'rb') as file:
        test_metadata = pickle.load(file)

    print('Test metadata is imported')

    test_dataset = FireDataset(test_metadata, train_dir=args.data_dir, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    best_model_bucket = args.best_model_bucket
    best_model_prefix = args.best_model_prefix
    best_model_file = args.best_model_file

    s3.download_file(best_model_bucket, f'{best_model_prefix}/{best_model_file}', best_model_file)

    tar_file_path = best_model_file

    try:
        with tarfile.open(tar_file_path, 'r:gz') as tar:
            tar.extractall(path='extracted_folder/')
        print(f'Tar file "{tar_file_path}" successfully extracted.')
    except Exception as e:
        print(f'Error extracting tar file "{tar_file_path}": {e}')

    print('Best Model is imported')

    model_name = tar_file_path.split('.')[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(f'extracted_folder/{model_name}.pth', map_location=device))
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

    result_file = args.result_file + model_name

    s3.put_object(Bucket=args.result_bucket, Key=f'{args.result_prefix}/{result_file}', Body=json.dumps(result))

    model_package_arn = args.model_package_arn

    sm.update_model_package(
    ModelPackageArn=model_package_arn,
    CustomerMetadataProperties={"acc": f"{acc}", "test_loss": f"{test_loss}"}
    )
    print('Results are registered to Model Registry')

    print('Job is Finished')
