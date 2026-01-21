"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os
import boto3
import sagemaker

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    model_package_group_name="TransportedLightGBM",
    pipeline_name="TransportedPipeline",
    base_job_prefix="transported",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    boto_session = boto3.Session(region_name=region)
    pipeline_session = PipelineSession(
        boto_session=boto_session,
        sagemaker_client=boto_session.client("sagemaker"),
        default_bucket=default_bucket,
    )

    if role is None:
        role = sagemaker.session.get_execution_role(pipeline_session)

    # -------- Parameters --------
    input_data = ParameterString(
        name="InputDataUrl",
        default_value="s3://choitore-data/inputdata/inputdata.csv",
    )

    accuracy_threshold = ParameterFloat(
        name="AccuracyThreshold",
        default_value=0.8,
    )

    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",
    )

    # -------- Preprocess --------
    processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.0-1",
            instance_type=processing_instance_type,
        ),
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
    )

    step_preprocess = ProcessingStep(
        name="PreprocessData",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/train", output_name="train"),
            ProcessingOutput(source="/opt/ml/processing/validation", output_name="validation"),
            ProcessingOutput(source="/opt/ml/processing/test", output_name="test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--input-data", input_data],
    )

    # -------- Train (LightGBM) --------
    image_uri = sagemaker.image_uris.retrieve(
        framework="lightgbm",
        region=region,
        version="3.3-2",
        py_version="py3",
        instance_type=training_instance_type,
    )

    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        output_path=f"s3://{default_bucket}/{base_job_prefix}/model",
        sagemaker_session=pipeline_session,
    )

    estimator.set_hyperparameters(
        objective="binary",
        metric="binary_error",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
    )

    step_train = TrainingStep(
        name="TrainLightGBM",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
            ),
            "validation": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri
            ),
        },
    )

    # -------- Evaluate --------
    evaluator = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="EvaluateModel",
        processor=evaluator,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/evaluation", output_name="evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )

    # -------- Register --------
    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        sagemaker_session=pipeline_session,
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"{step_eval.arguments['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']}/evaluation.json",
            content_type="application/json",
        )
    )

    step_register = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            model_package_group_name=model_package_group_name,
            approval_status=model_approval_status,
            model_metrics=model_metrics,
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
        ),
    )

    # -------- Condition --------
    cond = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="classification_metrics.accuracy.value",
        ),
        right=accuracy_threshold,
    )

    step_cond = ConditionStep(
        name="CheckAccuracy",
        conditions=[cond],
        if_steps=[step_register],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data,
            accuracy_threshold,
            model_approval_status,
        ],
        steps=[
            step_preprocess,
            step_train,
            step_eval,
            step_cond,
        ],
        sagemaker_session=pipeline_session,
    )

    return pipeline