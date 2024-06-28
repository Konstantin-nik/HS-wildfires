from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.functions import Join
from sagemaker.sklearn.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
import sagemaker


def get_evaluator_step(
    project: str,
    bucket_name: str,
    evaluator_instance_count: int,
    evaluator_instance_type: str,
    evaluation_image_uri: str,
    training_step,
    result_prefix: str,
    region: str,
):
    evaluation_processor = ScriptProcessor(
        role=sagemaker.get_execution_role(),
        image_uri=evaluation_image_uri,
        command=['python3'],
        instance_count=evaluator_instance_count,
        instance_type=evaluator_instance_type
    )

    return ProcessingStep(
        name=f"{project}-evaluation",
        processor=evaluation_processor,
        code="../model/evaluation.py",
        inputs=[
            ProcessingInput(
                source="s3://wildfires/data/test",
                destination="/opt/ml/processing/input",
            ),
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=Join(
                    on="/",
                    values=[f"s3://{bucket_name}", "evaluation_results"],
                ),
                output_name="evaluation_result",
            ),
            ProcessingOutput(
                source="/opt/ml/processing/output/model_arn.json",
                destination=Join(
                    on="/",
                    values=[f"s3://{bucket_name}", "models/last/model_arn.json"],
                ),
                output_name="model_arn",
            )
        ],
        job_arguments=[
            '--bucket', bucket_name,
            '--model_path', '/opt/ml/processing/model',
            '--data_path', '/opt/ml/processing/input',
            '--result_prefix', result_prefix,
            '--result_file', 'eva-result-',
            '--region', region
        ],
    )

