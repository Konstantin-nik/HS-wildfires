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
    evaluator_step
    
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
        code="../deployment/wait.py",
        inputs=[
            ProcessingInput(
                source=evaluator_step.properties.ProcessingOutputConfig.Outputs[
                    "model_arn"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            )
        ],
        job_arguments=[
            '--bucket_name', bucket_name,
            '--model_package_arn', '/opt/ml/processing/input/model_arn.json',
            '--region', region
        ],
    )
