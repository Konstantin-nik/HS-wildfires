import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker", "boto3"])

import os
import boto3
import sagemaker
import argparse
import pickle
import numpy as np
import json
import tarfile
import time



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--bucket_name', type=str, required=True)
    # parser.add_argument('--test_metadata_prefix', type=str, required=True)
    # parser.add_argument('--test_metadata_file', type=str, required=True)
    # parser.add_argument('--best_model_bucket', type=str, required=True)
    # parser.add_argument('--best_model_prefix', type=str, required=True)
    # parser.add_argument('--best_model_file', type=str, required=True)
    # parser.add_argument('--result_bucket', type=str, required=True)
    # parser.add_argument('--result_prefix', type=str, required=True)
    # parser.add_argument('--result_file', type=str, required=True)
    # parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_package_arn', type=str, required=True)
    parser.add_argument('--wait_time_seconds', type=int, required=False, default=300)
    parser.add_argument('--region', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    boto_session = boto3.Session()
    client = boto3.client('sagemaker', region_name=args.region)
    print(args.region)

    s3 = boto3.client('s3')

    
    print('Wait Step job is started')

    wait_time_seconds = args.wait_time_seconds 
    # time.sleep(wait_time_seconds)

    BUCKET_NAME = args.bucket_name
    model_package_arn = args.model_package_arn
    
    s3.download_file('wildfires', f'model_status/{model_package_arn}.json', 'status.json')

    with open('status.json', 'r') as file:
        status = json.load(file)

    approval_status = status['ApprovalStatus']

    while approval_status != "Approved":
        print('Model Approved Status is not Approved')
        time.sleep(wait_time_seconds)
        
        response = client.describe_model_package(ModelPackageName=model_package_arn)
        approval_status = response['ModelApprovalStatus']
        if approval_status == 'Rejected':
            print('Model Approved Status is Rejected')   
            break

    print('While Loop is Ended') 
    status['ApprovalStatus'] = approval_status
      

    with open('data.json', 'w') as file:
        json.dump(status, file)

    s3.upload_file(
        f'data.json',
        BUCKET_NAME,
        f"model_status/{model_package_arn}.json",
    )

    print('Model Approved Status is Changed')     
    
    print('Job is Finished')
