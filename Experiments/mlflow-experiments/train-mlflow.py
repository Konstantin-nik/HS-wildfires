import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

import mlflow
import sagemaker
# from smexperiments.tracker import Tracker

tracking_server_arn = os.environ.get('MLFLOW_TRACKING_URIs', None)
parent_run_id = os.environ.get('MLFLOW_PARENT_RUN_ID', None)
mlflow_experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', None)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val_dir', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    return parser.parse_args()

def load_data(train_dir, val_dir):
    X_train = np.load(os.path.join(train_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(val_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(val_dir, 'y_val.npy'))

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader


def train(args):
    # nested = False

    # if parent_run_id:
    #     nested = True

    train_loader, val_loader = load_data(args.train_dir, args.val_dir)

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    # model = model.cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print('# Device', device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    with mlflow.start_run(run_name=sagemaker.utils.name_from_base(args.run_name)) as run:
        mlflow.autolog()

        mlflow.log_params({
            'epochs': args.epochs,
            'batch_size': args.batch_size
        }, run_id=run.info.run_id)

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images = images.permute(0, 3, 1, 2).to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Log training loss
            # tracker.log_metric('training_loss', running_loss / len(train_loader), step=epoch)
            mlflow.log_metric('training_loss', running_loss / len(train_loader), step=epoch, run_id=run.info.run_id)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.permute(0, 3, 1, 2).to(device)
                    labels = labels.float().unsqueeze(1).to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()

            # Log validation loss
            # tracker.log_metric('validation_loss', val_loss / len(val_loader), step=epoch)
            mlflow.log_metric('validation_loss', val_loss / len(val_loader), step=epoch, run_id=run.info.run_id)

            print(f'Epoch {epoch+1}/{args.epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')

        torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

        mlflow.log_artifact(os.path.join(args.model_dir, 'model.pth'))

        mlflow.end_run(status='FINISHED')


if __name__ == '__main__':
    args = parse_args()

    print('# ENAME', mlflow_experiment_name)
    print('# RNUM', args.run_name)

    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(mlflow_experiment_name)
    
    # with mlflow.start_run(run_id=parent_run_id):
    train(args)import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

import mlflow
import sagemaker
# from smexperiments.tracker import Tracker

tracking_server_arn = os.environ.get('MLFLOW_TRACKING_URIs', None)
parent_run_id = os.environ.get('MLFLOW_PARENT_RUN_ID', None)
mlflow_experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', None)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val_dir', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    return parser.parse_args()

def load_data(train_dir, val_dir):
    X_train = np.load(os.path.join(train_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(val_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(val_dir, 'y_val.npy'))

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader


def train(args):
    # nested = False

    # if parent_run_id:
    #     nested = True

    train_loader, val_loader = load_data(args.train_dir, args.val_dir)

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    # model = model.cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print('# Device', device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    with mlflow.start_run(run_name=sagemaker.utils.name_from_base(args.run_name)) as run:
        mlflow.autolog()

        mlflow.log_params({
            'epochs': args.epochs,
            'batch_size': args.batch_size
        }, run_id=run.info.run_id)

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images = images.permute(0, 3, 1, 2).to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Log training loss
            # tracker.log_metric('training_loss', running_loss / len(train_loader), step=epoch)
            mlflow.log_metric('training_loss', running_loss / len(train_loader), step=epoch, run_id=run.info.run_id)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.permute(0, 3, 1, 2).to(device)
                    labels = labels.float().unsqueeze(1).to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()

            # Log validation loss
            # tracker.log_metric('validation_loss', val_loss / len(val_loader), step=epoch)
            mlflow.log_metric('validation_loss', val_loss / len(val_loader), step=epoch, run_id=run.info.run_id)

            print(f'Epoch {epoch+1}/{args.epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')

        torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

        mlflow.log_artifact(os.path.join(args.model_dir, 'model.pth'))

        mlflow.end_run(status='FINISHED')


if __name__ == '__main__':
    args = parse_args()

    print('# ENAME', mlflow_experiment_name)
    print('# RNUM', args.run_name)

    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(mlflow_experiment_name)
    
    # with mlflow.start_run(run_id=parent_run_id):
    train(args)