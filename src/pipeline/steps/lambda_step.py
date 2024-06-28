from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput
from sagemaker.lambda_helper import Lambda


def get_lambda_step(
    project: str,
    bucket_name: str,
    process_instance_count_param: int,
    process_instance_type_param: str,
    evaluation_image_uri: str,
    region: str,

    lambda_check_function_arn: str,
    step_suffix: str
):
    print("Starting get_lambda_step")
    func = Lambda(function_arn=lambda_check_function_arn)

    # Lambda Step: Model approval status sorgulama
    return LambdaStep(
        name=f"CheckApprovalStatus{step_suffix}",
        lambda_func=func,
        inputs={"model_package_arn": f"models/last/model_arn.json"},
        outputs=[
            LambdaOutput(output_name='Payload')
        ]
    )