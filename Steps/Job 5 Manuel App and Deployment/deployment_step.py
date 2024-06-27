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

    # parser.add_argument('--test_metadata_bucket', type=str, required=True)
    # parser.add_argument('--test_metadata_prefix', type=str, required=True)
    # parser.add_argument('--test_metadata_file', type=str, required=True)
    # parser.add_argument('--best_model_bucket', type=str, required=True)
    # parser.add_argument('--best_model_prefix', type=str, required=True)
    # parser.add_argument('--best_model_file', type=str, required=True)
    # parser.add_argument('--result_bucket', type=str, required=True)
    # parser.add_argument('--result_prefix', type=str, required=True)
    # parser.add_argument('--result_file', type=str, required=True)
    # parser.add_argument('--data_dir', type=str, required=True)
    # parser.add_argument('--model_package_arn', type=str, required=True)
    # parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--region', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    boto_session = boto3.Session()
    region = boto_session.region_name

    print(region)
    print(args.region)

    sm = boto3.client('sagemaker', region_name='eu-central-1')

    s3 = boto3.client('s3')
    print('Deployment job is started')



    
    


    print('Job is Finished')
