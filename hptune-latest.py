from kfp import dsl, compiler
from kfp.dsl import pipeline
from google.cloud import aiplatform
from google_cloud_pipeline_components.v1 import hyperparameter_tuning_job
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google_cloud_pipeline_components.v1.model_evaluation import ModelEvaluationClassificationOp
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.hyperparameter_tuning_job import (
    HyperparameterTuningJobRunOp,
)
import get_tuning_job_results

METRIC_SPEC = dict(logloss="minimize")

@pipeline(
    name="xgboost-hypertune-pipeline",
    description="Pipeline for XGBoost hyperparameter tuning using cloudml-hypertune.",
)
def xgboost_hpt_pipeline(
    project: str,
    location: str,
    root_dir: str,
    container_image_uri: str,
    data_source_bigquery_table_path: str,
    target_column: str,
    learning_rate_min: float = 0.01,
    learning_rate_max: float = 0.3,
    max_trial_count: int = 10,
    parallel_trial_count: int = 2,
    machine_type: str = "n1-standard-4",
):
    # # Step 1: Define Worker Pool Specs
    # worker_pool_specs = [
    #     {
    #         "machine_spec": {
    #             "machine_type": machine_type,
    #         },
    #         "replica_count": 1,
    #         "container_spec": {
    #             "image_uri": container_image_uri,
    #             "command": [
    #                 "python3",
    #                 "train.py",
    #             ],
    #             "args": [
    #                 "--data_source_bigquery_table_path",
    #                 data_source_bigquery_table_path,
    #                 "--target_column",
    #                 target_column,
    #                 "--learning_rate",
    #                 "{{$.trial.parameters.learning_rate}}",
    #                 "--model_output_path",
    #                 "{{$.output.model}}",
    #             ],
    #         },
    #     },
    # ]

    # Step 2: Hyperparameter Tuning Job

    # Hyperparameter Tuning Job
    hpt_job = HyperparameterTuningJobRunOp(
        project=project,
        location=location,
        base_output_directory=root_dir,
        display_name="xgboost-binary-classification-hpt",
        study_spec_metrics=[
            {
                "metric_id": "logloss",  # Matches hypertune metric tag
                "goal": "MINIMIZE",
            }
        ],
        study_spec_parameters=[
            {
                "parameter_id": "learning_rate",
                "double_value_spec": {
                    "min_value": learning_rate_min,
                    "max_value": learning_rate_max,
                },
                "scale_type": "UNIT_LINEAR_SCALE",
            },
        ],
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": machine_type,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": container_image_uri,
                    "command": [
                        "python3",
                        "train.py",
                    ],
                    "args": [
                        "--data_source_bigquery_table_path",
                        data_source_bigquery_table_path,
                        "--target_column",
                        target_column,
                        "--learning_rate",
                        "{{$.trial.parameters.learning_rate}}",
                        "--model_output_path",
                        "{{$.output.model}}",
                    ],
                },
            },
        ],
    )
    
    print(hpt_job.outputs)

    hypertune_results_step = get_tuning_job_results.get_hyperparameter_tuning_results(
        project=project,
        location=location,
        job_resource=hpt_job.output,
        study_spec_metrics=hyperparameter_tuning_job.utils.serialize_metrics(
            METRIC_SPEC
        ),
    ).set_display_name("retreive-tuning-job-results")


    

    # # Step 3: Upload the Best Model
    # upload_model_task = ModelUploadOp(
    #     project=project,
    #     location=location,
    #     display_name="xgboost-binary-classification-model",
    #     artifact_uri=best_trail_op.outputs,
    #     serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
    # ).after(best_trail_op)

    # # Step 4: Batch Prediction
    # batch_prediction_task = ModelBatchPredictOp(
    #     project=project,
    #     location=location,
    #     model=upload_model_task.outputs["model"],
    #     gcs_source=data_source_bigquery_table_path,
    #     gcs_destination_prefix=f"{root_dir}/batch_predictions",
    #     machine_type=machine_type,
    # )

    # # Step 5: Model Evaluation
    # evaluation_task = ModelEvaluationClassificationOp(
    #     project=project,
    #     location=location,
    #     target_field_name=target_column,
    #     prediction_gcs_source=batch_prediction_task.outputs["gcs_output_directory"],
    #     prediction_gcs_format="jsonl",
    # )

    # # Outputs
    # evaluation_task.set_display_name("Model Evaluation Metrics")
    # upload_model_task.set_display_name("Upload Best Model")




compiler.Compiler().compile(
    pipeline_func=xgboost_hpt_pipeline,
    package_path="xgboost_hypertune_pipeline.json",
)



# Initialize Vertex AI
aiplatform.init(
    project="mdepew-assets",
    location="us-central1",
    staging_bucket="gs://testing-hptune-pipeline-depew/xgboost-pipeline",
)

# Run the pipeline
pipeline_job = aiplatform.PipelineJob(
    display_name="xgboost-hypertune-pipeline",
    template_path="xgboost_hypertune_pipeline.json",
    parameter_values={
        "project": "mdepew-assets",
        "location": "us-central1",
        "root_dir": "gs://testing-hptune-pipeline-depew/xgboost-pipeline",
        "container_image_uri": "",
        "data_source_bigquery_table_path": "bq://mdepew-assets.synthetic.synthetic_mortgage_data",
        "target_column": "refinance",
        "learning_rate_min": 0.01,
        "learning_rate_max": 0.3,
        "max_trial_count": 10,
        "parallel_trial_count": 2,
        "machine_type": "n1-standard-4",
    },
)

# Submit the job
pipeline_job.run(sync=True)