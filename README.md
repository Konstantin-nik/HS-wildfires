# WildFire Detection in SageMaker

In this project, SageMaker and its services are used to create a pipeline for image processing, training a deep learning model, and subsequently deploying it to predict whether images contain fire, specifically in forest areas or vegetation.

The objective is to use the endpoint to detect fire in streaming security cameras. To achieve this, the stream can be separated into frames, and each frame is analyzed to detect the presence of fire.

## Code

The code presented here was developed for SageMaker, and its proper testing should be done within the JupyterLab service.

## Data

We use open datasets found on the internet:

* [Kaggle Fire Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
* [Roboflow Fire Project 1](https://universe.roboflow.com/fire-iak32/fire-project/dataset/4)
* [Roboflow Fire Project 2](https://universe.roboflow.com/fire-dataset-tp9jt/fire-detection-sejra/)

These datasets can be downloaded for classification tasks. Currently, we **manually inject data** into the pipeline. Using the S3 service, the data is stored in the S3 bucket **`wildfire/download`** into the following folders:

* `wildfire/download/fire_image`: This folder contains images classified with fire.
* `wildfire/download/no_fire_image`: This folder contains images classified with no fire.

This copying process is done **manually through the AWS UI or using the terminal**, as this will be all the data to be used for training the models.

## Pipeline

Each `Job #` folder is a step in the pipeline. Each one has a jupuyter noteboook that calls a job inside sagemakcer to perform the task descirbed in the folder name:

* *Job 1 preprocess*: Data is taken from the s2 bucket, trasfromed in size and format and ingest it to the feature store (FS)
* *Job 2 training*: A Resnet is trained uisng the data from the FS in a training job.
