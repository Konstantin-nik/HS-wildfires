import json
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    logger.info("Lambda 'lambda_check_status' started")
    client = boto3.client('sagemaker')
    model_package_arn = event['model_package_arn']
    s3 = boto3.client('s3')
    BUCKET_NAME = 'wildfires'

    response = client.describe_model_package(ModelPackageName=model_package_arn)
    status = response['ModelApprovalStatus']

    logger.info(f"Approval status for model package {model_package_arn}: {status}")
    with open("model_status.json" , mode="w") as f:
        json.dumps({'ApprovalStatus': status}, f)
    s3.upload_file(
        f'model_status.json',
        BUCKET_NAME,
        f"model_status/{model_package_arn}.json",
    )

    return {
        'statusCode': 200,
        'body': json.dumps({'ApprovalStatus': status})
    }











