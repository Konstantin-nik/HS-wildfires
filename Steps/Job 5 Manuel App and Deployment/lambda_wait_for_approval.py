import json
import boto3
import logging
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info("Lambda 'lambda_wait_for_approval' started")
    wait_time_seconds = 300  # 3 dakika bekle
    time.sleep(wait_time_seconds)
    logger.info(f"Waiting for {wait_time_seconds} seconds completed")
    return event  # Aynı event'i geri döndür
