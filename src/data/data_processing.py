import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker", "boto3"])

import os
import json
import uuid
import time
import boto3
import logging
import argparse
import sagemaker
import pandas as pd

from datetime import datetime
from logging.handlers import MemoryHandler
from botocore.exceptions import ClientError 
from sklearn.model_selection import train_test_split
from sagemaker.feature_store.feature_group import FeatureGroup, FeatureDefinition, FeatureTypeEnum


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("data_processor")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_bucket', type=str, required=True)
    parser.add_argument('--src_prefix', type=str, required=True)
    parser.add_argument('--dest_bucket', type=str, required=True)
    parser.add_argument('--dest_prefix', type=str, required=True)
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--seed', type=int, required=False, default=1)
    parser.add_argument('--output_path', type=str, required=True)

    args = parser.parse_args()
    return args


def copy_s3_objects(src_bucket, src_prefix, dest_bucket, dest_prefix):
    logger.info('Coping data from download/ to data/raw_data/')
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=src_bucket, Prefix=src_prefix)
    logger.info('Objects were listed')

    for obj in response.get('Contents', []):
        copy_source = {'Bucket': src_bucket, 'Key': obj['Key']}
        dest_key = os.path.join(dest_prefix, os.path.relpath(obj['Key'], src_prefix))
        s3.copy_object(CopySource=copy_source, Bucket=dest_bucket, Key=dest_key)
        logger.info(f'Copied {obj["Key"]} to {dest_key}')


def update_fs_version(version, session):
    local_file_path = "/tmp/fs_version.txt"
    with open(local_file_path, mode="w") as file:
        file.write(str(version))

    session.upload_data(path=local_file_path, bucket=bucket_name, key_prefix="feature-store")
    logger.info("FS version updated")


def get_fs_last_vesrion(s3):
    version_file_path = "feature-store/fs_version.txt"
    local_file_path = "/tmp/fs_version.txt"
    try:
        s3.download_file(bucket_name, version_file_path, local_file_path)
        logger.info(f"Version file successfully loaded: {local_file_path}")
        with open(local_file_path) as file:
            version = file.readline().strip()
        return int(version)
    except ClientError as e:
        logger.info(f"No version file founded")
        return 0


def wait_for_group_created(feature_group_name, sagemaker_client):
    while True:
        response = sagemaker_client.describe_feature_group(FeatureGroupName=feature_group_name)
        status = response['FeatureGroupStatus']
        if status == 'Created':
            logger.info('Feature group created successfully.')
            break
        elif status == 'CreateFailed':
            raise Exception(
                f'Failed to create feature group: {response["FailureReason"]}'
            )
        else:
            logger.info('Waiting for feature group to be created...')
            time.sleep(5)


def create_feature_group(s3, sm_role, sm_session, sagemaker_client):
    fs_version = get_fs_last_vesrion(s3) + 1
    feature_group_name = f"wildfire-feature-group-v{fs_version}"
    update_fs_version(fs_version, sm_session)

    # Define feature definitions
    feature_definitions = [
        FeatureDefinition('image_id', FeatureTypeEnum.STRING),
        FeatureDefinition('image_location', FeatureTypeEnum.STRING),
        FeatureDefinition('label', FeatureTypeEnum.INTEGRAL),
        FeatureDefinition('image_type', FeatureTypeEnum.STRING),
        FeatureDefinition('event_time', FeatureTypeEnum.STRING),
        FeatureDefinition('data_purpose', FeatureTypeEnum.STRING), # train or test
    ]

    # Create the feature group
    feature_group = FeatureGroup(
        name=feature_group_name,
        sagemaker_session=sm_session,
        feature_definitions=feature_definitions
    )

    # Run feature group create
    feature_group.create(
        s3_uri=f's3://{bucket_name}/feature-store',
        description=f'Feature group for storing fire image features. Version v{fs_version}',
        tags={'Version': f"v{fs_version}"},
        record_identifier_name='image_id',
        event_time_feature_name='event_time',
        role_arn=sm_role
    )

    wait_for_group_created(feature_group_name, sagemaker_client)

    return feature_group


def list_s3_objects(s3, prefix):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    return [obj['Key'] for obj in response.get('Contents', [])]


def generate_metadata(image_list, label):
    metadata = []
    for image_location in image_list:
        image_id = str(uuid.uuid4())
        image_type = image_location.split('.')[-1]
        metadata.append({
            'image_id': image_id,
            'image_location': f's3://{bucket_name}/{image_location}',
            'label': label,
            'image_type': image_type,
            'event_time': datetime.utcnow().isoformat() + 'Z'
        })
    return metadata


def get_metadata(s3):
    # List images
    fire_images = list_s3_objects(s3, 'data/raw_data/fire_images/')
    non_fire_images = list_s3_objects(s3, 'data/raw_data/non_fire_images/')

    # Generate metadata
    fire_metadata = generate_metadata(fire_images, 1)
    non_fire_metadata = generate_metadata(non_fire_images, 0)
    all_metadata = fire_metadata + non_fire_metadata

    return all_metadata


def update_metadata(s3, metadata, data_purpose):
    for record in metadata:
        record['data_purpose'] = data_purpose

        image_location = record['image_location']
        bucket, key = image_location.replace('s3://', '').split('/', 1)
        copy_source = {
            'Bucket': bucket,
            'Key': key
        }
        image_id = record['image_id']
        image_type = record['image_type']
        new_image_location = f'data/{data_purpose}/{image_id}.{image_type}'

        s3.copy(copy_source, bucket_name, new_image_location)
        logger.info(f'Copyied {image_location} to {new_image_location}')
        record['image_location'] = f's3://{bucket_name}/{new_image_location}'

    return metadata


def convert_to_df(metadata):
    return pd.DataFrame(metadata)


if __name__ == "__main__":
    args = parse_args()

    s3 = boto3.client('s3')
    sm_session = sagemaker.Session(boto3.Session(region_name=args.region))
    sm_role = sagemaker.get_execution_role(sm_session)
    sagemaker_client = sm_session.boto_session.client('sagemaker')
    feature_store_client = sm_session.boto_session.client('sagemaker-featurestore-runtime')

    bucket_name = 'wildfires'

    print("Getting data from download/")
    copy_s3_objects(args.src_bucket, args.src_prefix, args.dest_bucket, args.dest_prefix)
    print("Data was copied to raw_data!")

    print("Creating Feature Group")
    feature_group = create_feature_group(s3, sm_role, sm_session, sagemaker_client)

    print("Getting metadata")
    metadata = get_metadata(s3)

    print("Splitting metadata")
    train_metadata, test_metadata = train_test_split(
        metadata,
        test_size=0.2,
        stratify=[m['label'] for m in metadata],
        random_state=args.seed
    )

    print("Updating metadata and files")
    train_metadata = update_metadata(s3, train_metadata, 'train')
    test_metadata = update_metadata(s3, test_metadata, 'test')

    all_metadata = train_metadata + test_metadata

    df = convert_to_df(all_metadata)

    print("Ingesting metadata to feature storage")
    feature_group.ingest(data_frame=df, max_workers=3, wait=True)

    for handler in logger.handlers:
        if isinstance(handler, MemoryHandler):
            handler.flush()

    print("Job finished!")