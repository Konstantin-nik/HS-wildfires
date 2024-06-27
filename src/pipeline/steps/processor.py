import sagemaker

from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import ScriptProcessor


def get_processor_step(
    project: str,
    bucket_name: str,
    process_instance_count_param: int,
    process_instance_type_param: str,
    sklearn_image_uri: str,
    region: str,
    seed: int
):
    processing_processor = ScriptProcessor(
        role=sagemaker.get_execution_role(),
        image_uri=sklearn_image_uri,
        command=['python3'],
        instance_count=process_instance_count_param,
        instance_type=process_instance_type_param
    )

    return ProcessingStep(
        name=f"{project}-processing",
        processor=processing_processor,
        code="../data/data_processing.py",
        job_arguments=[
            '--src_bucket', bucket_name,
            '--src_prefix', 'download/',
            '--dest_bucket', bucket_name,
            '--dest_prefix', 'data/raw_data/',
            '--region', region,
            '--seed', seed
        ],
    )