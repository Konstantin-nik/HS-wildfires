import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker", "boto3"])

import boto3
import sagemaker
import argparse
import json
from sagemaker.pytorch import PyTorchModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--bucket_name', type=str, required=True)
    # parser.add_argument('--test_metadata_prefix', type=str, required=True)
    # parser.add_argument('--test_metadata_file', type=str, required=True)
    # parser.add_argument('--best_model_bucket', type=str, required=True)
    parser.add_argument('--model_prefix', type=str, required=True)
    parser.add_argument('--model_filename', type=str, required=True)
    # parser.add_argument('--result_bucket', type=str, required=True)
    # parser.add_argument('--result_prefix', type=str, required=True)
    # parser.add_argument('--result_file', type=str, required=True)
    # parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_package_arn', type=str, required=True)
    # parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--region', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session)

    print(args.region)

    s3 = boto3.client('s3')
    role = sagemaker.get_execution_role(sagemaker_session=sm_session)

    print('Deployment job is started')

    BUCKET_NAME = args.bucket_name

    with open(args.model_package_arn, 'r') as file:
        model_arn = json.load(file)
    model_package_arn = model_arn['ARN']

    model_filename = args.model_filename
    model_prefix = args.model_prefix
    model_path = f"{model_prefix}/{model_filename}.tar.gz"

    s3.download_file('wildfires', f'model_status/{model_package_arn}.json', 'status.json')

    with open('status.json', 'r') as file:
        status = json.load(file)

    status['ApprovalStatus'] = "Deployed"

    with open('data.json', 'w') as file:
        json.dump(status, file)

    s3.upload_file(
        'data.json',
        BUCKET_NAME,
        f"model_status/{model_package_arn}.json",
    )

    print('Model Status is changed')

    print('Model Deployment is started')

    model = PyTorchModel(
        model_data=f"s3://{BUCKET_NAME}/{model_path}",
        role=role,
        framework_version='2.0.0',
        py_version='py310',
        entry_point='inference.py',
        sagemaker_session=sm_session
    )

    model.deploy(
        initial_instance_count=1,
        instance_type='ml.g5.xlarge'
    )

    print('Model Deployment is deployed')

    print('Job is Finished')
