from sagemaker.processing import ProcessingOutput, ProcessingInput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import ScriptProcessor
from sagemaker.workflow.functions import Join
import sagemaker


def get_wait_step(
    project: str,
    bucket_name: str,
    process_instance_count_param: int,
    process_instance_type_param: str,
    evaluation_image_uri: str,
    region: str,
    
    # test_metadata_prefix: str,
    # best_model_prefix: str,
    # test_metadata_file: str,
    # best_model_file: str,
    # result_prefix: str,
    # data_dir: str,
    model_package_arn: str
    # wait_time_seconds: int
):
    evaluation_processor = ScriptProcessor(
        role=sagemaker.get_execution_role(),
        image_uri=evaluation_image_uri,
        command=['python3'],
        instance_count=process_instance_count_param,
        instance_type=process_instance_type_param
    )

    return ProcessingStep(
        name=f"{project}-wait",
        processor=evaluation_processor,
        code="wait_step.py",
        job_arguments=[
            '--bucket_name', bucket_name,
            # '--best_model_bucket', bucket_name,
            # '--test_metadata_prefix', test_metadata_prefix,
            # '--best_model_prefix', best_model_prefix,
            # '--test_metadata_file', test_metadata_file,
            # '--best_model_file', best_model_file,
            # '--result_bucket', bucket_name,
            # '--result_prefix', result_prefix,
            # '--result_file', 'eva-result-',
            # '--data_dir', data_dir,
            '--model_package_arn', model_package_arn,
            # '--wait_time_seconds', wait_time_seconds,
            '--region', region
        ],
    )
