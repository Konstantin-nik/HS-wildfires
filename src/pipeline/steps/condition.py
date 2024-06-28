from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.fail_step import FailStep
import boto3
import sagemaker


def get_conditional_step(
    project: str,
    bucket_name: str,
    process_instance_count_param: int,
    process_instance_type_param: str,
    evaluation_image_uri: str,
    region: str,

    model_path: str,
    wait_step: ProcessingStep,
    deployment_step: ProcessingStep,
    condition_step_suffix: str
):
    print("Starting conditional_step")

    step_fail = FailStep(
        name=f"{project}-fail",
        error_message="Execution failed due to ApproveStatus"
    )

    boto_session = boto3.Session(region_name=region)
    sm_session = sagemaker.Session(boto_session)

    s3 = boto3.client('s3')

    s3.download_file('wildfires', 'models/last/model_arn.json', 'model_arn.json')

    with open('model_arn.json', 'r') as file:
        model_arn = json.load(file)

    model_package_arn = model_arn['ARN']

    return ConditionStep(
        name=f"CheckIfApproved{condition_step_suffix}",
        conditions=[
            ConditionEquals(
                left=JsonGet(
                    step=wait_step,
                    s3_uri=Join(
                            on="/",
                            values=[
                                "s3://wildfires",
                                "model_status",
                                f"{model_package_arn}.json"
                            ]
                    ),
                    json_path="$.ApprovalStatus"
                ),
                right="Approved"
            )
        ],
        if_steps=[deployment_step],
        else_steps=[step_fail]
    )