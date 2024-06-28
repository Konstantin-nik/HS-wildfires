import sagemaker

from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep


def get_trainer_step(
    project: str,
    bucket_name: str,
    tracking_server_arn: str,
    train_instance_count: int,
    train_instance_type: str,
    region: str,
    epochs_num: int,
    batch_size: int,
    learning_rate: float,
    seed: int
):
    run_name = 'train-resnet-fire'
    estimator = PyTorch(
        role=sagemaker.get_execution_role(),
        entry_point="../model/training.py",
        instance_count=train_instance_count,
        instance_type=train_instance_type,
        input_mode='File',
        py_version="py39",
        framework_version="1.13",
        environment={
            'MLFLOW_TRACKING_URIs': tracking_server_arn,
            'MLFLOW_EXPERIMENT_NAME': f"{project}-training-pipeline",
        },
        dependencies=['requirements.txt'],
        hyperparameters={
            'num-epochs': epochs_num,
            'batch-size': batch_size,
            'learning-rate': learning_rate,
            'run-name': run_name,
            'bucket': bucket_name,
            'region': region,
            'train_dir': '/opt/ml/input/data/data_input',
            'seed': seed
        },
        output_path=f's3://{bucket_name}/models/'
    )

    data_input = TrainingInput(
        s3_data="s3://wildfires/data/train/",
        input_mode='File'
    )

    return TrainingStep(
        name=f"{project}-training",
        estimator=estimator,
        inputs={"data_input": data_input},
    )