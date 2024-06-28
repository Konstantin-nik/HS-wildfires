import os
import boto3
import sagemaker
import torch
import argparse
import mlflow

import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision import models
from torchvision import transforms
from botocore.exceptions import ClientError
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sagemaker.feature_store.feature_group import FeatureGroup

tracking_server_arn = os.environ.get('MLFLOW_TRACKING_URIs', None)
mlflow_experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', None)

model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
model_path = os.path.join(model_dir, 'model.pth')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--run-name', type=str, default="")
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--seed', type=int, required=False, default=1)

    args = parser.parse_args()

    return args


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
        # print("Train dir:", self.train_dir, "\nImage Path:", img_path)
        img_path = os.path.join(self.train_dir, img_path)
        image = Image.open(img_path).convert('RGB')
        label = self.metadata[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label


def train(model, train_loader, optimizer, loss_function,
          epoch, device, run, checkpoint_dir):
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
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(state, f"{checkpoint_dir}/epoch_{epoch}.pth")
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)

    mlflow.log_metric('training_loss', train_loss,
                      step=epoch, run_id=run.info.run_id)


def validation(model, val_loader, loss_function, epoch, device, run):
    model = model.to(device)
    loss_function = loss_function.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)

    mlflow.log_metric('test_loss', test_loss,
                      step=epoch, run_id=run.info.run_id)
    mlflow.log_metric('test_acc', correct / len(val_loader.dataset),
                      step=epoch, run_id=run.info.run_id)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    return test_loss


def argmin(lst):
    if not lst:
        raise ValueError("argmin() arg is an empty sequence")

    min_index = 0
    min_value = lst[0]

    for i in range(1, len(lst)):
        if lst[i] < min_value:
            min_value = lst[i]
            min_index = i

    return min_index


def train_validation(model, optimizer, criterion,
                     train_loader, val_loader, checkpoint_dir,
                     device, run, n_epochs=1):
    losses_list = []

    for epoch in range(n_epochs):
        train(model, train_loader, optimizer, criterion,
              epoch, device, run, checkpoint_dir)
        losses_list.append(
            validation(model, val_loader, criterion, epoch, device, run)
        )

    best_epoch_num = argmin(losses_list)
    print("Best epoch is", best_epoch_num)
    path_to_best_epoch = f"{checkpoint_dir}/epoch_{best_epoch_num}.pth"
    best_epoch_state = torch.load(path_to_best_epoch)
    return best_epoch_state["model_state"]


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

    print("train_dir:", args.train_dir)

    print("Setting up tracking and Experiment")
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(mlflow_experiment_name)

    print('####', tracking_server_arn, mlflow_experiment_name)

    print("Loading metadata")
    # ----------------------------------------------

    feature_group_name = f'wildfire-feature-group-v{get_fs_last_vesrion(s3, args.bucket)}'

    feature_group = FeatureGroup(
        name=feature_group_name, sagemaker_session=sm_session
    )

    fs_train_query = feature_group.athena_query()

    query = f"""
    SELECT *
    FROM "{fs_train_query.table_name}"
    WHERE data_purpose = 'train';
    """

    response = get_responce_athena(query, athena_client)
    metadata = get_metadata(response)
    train_metadata, val_metadata = train_test_split(
        metadata, test_size=0.25,
        stratify=[m['label'] for m in metadata],
        random_state=args.seed
    )

    print('## Metadata Loaded')
    print(f'{len(metadata)} Metadata loaded')
    # -----------------------------------------------

    print("Initializing model")
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print("Model initialized")

    model_name = f'model_resnet18_bs{args.batch_size}_lr{args.learning_rate}_ne{args.num_epochs}_s{args.seed}.pth'
    model_path = f'/opt/ml/model/{model_name}'
    checkpoint_dir = '/opt/ml/checkpoint'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print("Start training")
    with mlflow.start_run(run_name=sagemaker.utils.name_from_base(args.run_name)) as run:
        mlflow.autolog()

        mlflow.log_params({
            'epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'train size': len(train_metadata),
            'validation size': len(val_metadata),
            'learning rate': args.learning_rate,
        }, run_id=run.info.run_id)

        best_model_state = train_validation(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=checkpoint_dir,
            device=device,
            n_epochs=args.num_epochs,
            run=run,
        )

        torch.save(best_model_state, model_path)
        torch.save(best_model_state, f'/opt/ml/model/model.pth')
        mlflow.pytorch.log_model(model, artifact_path="models")

        mlflow.end_run(status='FINISHED')
