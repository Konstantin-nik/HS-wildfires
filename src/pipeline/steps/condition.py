from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.fail_step import FailStep


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
    model_package_arn: str,
    condition_step_suffix: str
):
    print("Starting conditional_step")

    step_fail = FailStep(
        name=f"{project}-fail",
        error_message="Execution failed due to ApproveStatus"
    )

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