import sagemaker
from sagemaker.workflow.steps import LambdaStep, ModelStep, ConditionStep
from sagemaker.workflow.condition_step import JsonGet
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ParameterString
from sagemaker.pytorch import PyTorchModel


def get_manuel_approve_step(
    project: str,
    bucket_name: str,
    process_instance_count_param: int,
    process_instance_type_param: str,
    evaluation_image_uri: str,
    region: str,
    model_path: str,
    lambda_function_arn: str,
    model_package_arn: str
):

    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    # Pipeline parametreleri
    model_package_arn = model_package_arn

    # Lambda Step: Model approval status sorgulama
    lambda_step = LambdaStep(
        name="CheckApprovalStatus",
        lambda_function_arn="arn:aws:lambda:us-west-2:123456789012:function:check-approval-status",
        inputs={"model_package_arn": model_package_arn},
        output_path="approval_status"
    )

    # Condition Step: Model approval status kontrol etme
    return ConditionStep(
        name="CheckIfApproved",
        conditions=[
            ConditionEquals(
                left=JsonGet(
                    step_name=lambda_step.name,
                    property_file="approval_status",
                    json_path="$.ApprovalStatus"
                ),
                right="Approved"
            )
        ],
        if_steps=[
            # Model deploy adımı
            ModelStep(
                name="DeployModel",
                model=PyTorchModel(
                    model_data=f"s3://{bucket_name}/{model_path}",
                    role=role,
                    framework_version='2.0.0',
                    py_version='py310',
                    entry_point='inference.py'
                ),
                instance_count=1,
                instance_type="ml.g5.xlarge",
            )
        ],
        else_steps=[
            # Model approval status bekleme adımı
            LambdaStep(
                name="WaitForApproval",
                lambda_function_arn="arn:aws:lambda:us-west-2:123456789012:function:wait-for-approval",
                inputs={"model_package_arn": model_package_arn},
                output_path="approval_status"
            )
        ]
    )