import sagemaker
import argparse
from sagemaker.pytorch import PyTorchModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', type=str, default=None)
    parser.add_argument('--model_filename', type=str, default=None)
    parser.add_argument('--model_package_group_name', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    args = parse_args()
    BUCKET_NAME = args.bucket_name
    model_filename = args.model_filename
    model_path = f"models/{model_filename}.tar.gz"

    env = {'MMS_DEFAULT_RESPONSE_TIMEOUT': '1000000'}  # Set the timeout to 500 seconds or higher

    model = PyTorchModel(
        model_data=f"s3://{BUCKET_NAME}/{model_path}",
        role=role,
        framework_version='2.0.0',
        py_version='py310',
        entry_point='inference.py',
        env=env
    )

    # Model Package
    model_package = model.register(
        content_types=['application/json'],    # Input type JSON
        response_types=['application/json'],   # Output type JSON
        inference_instances=['ml.g5.xlarge'],  # Inference instance type
        transform_instances=['ml.g5.xlarge'],  # Transform instance type
        model_package_group_name=args.model_package_group_name  # Model Group Name
    )
    print('Model package ARN:', model_package.model_package_arn)