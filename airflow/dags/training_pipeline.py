from __future__ import annotations
import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.pipeline.training_pipeline import TrainingPipeline

run_pipeline = TrainingPipeline()

with DAG(
    dag_id="Gemstone_Training_Pipeline",
    default_args={"retries": 2},
    description="It is Training Pipeline",
    schedule="@weekly", # Every week
    start_date=pendulum.datetime(2025, 5, 15, tz="UTC"),
    catchup=False,
    tags=["machine_learning", "regression", "gemstone"]
) as dag:

    dag.doc_md = __doc__

    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        train_data_path, test_data_path = run_pipeline.run_data_ingestion()
        ti.xcom_push(
            key="data_ingestion_artifact",
            value={
                "train_data_path": train_data_path,
                "test_data_path": test_data_path
            }
        )

    def data_transformations(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_artifact = ti.xcom_pull(
            task_ids="data_ingestion", key="data_ingestion_artifact"
        )
        train_arr, test_arr = run_pipeline.run_data_transformation(data_ingestion_artifact)
        ti.xcom_push(
            key="data_transformation_artifact",
            value={
                "train_arr": train_arr.tolist(),
                "test_arr": test_arr.tolist()
            }
        )

    def model_trainer(**kwargs):
        import numpy as np
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(
            task_ids="data_transformations", key="data_transformation_artifact"
        )
        train_arr = np.array(data_transformation_artifact["train_arr"])
        test_arr = np.array(data_transformation_artifact["test_arr"])
        run_pipeline.run_model_training(train_arr, test_arr)

    data_ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
    )
    data_ingestion_task.doc_md = dedent("Ingestion of training and testing data")

    data_transformations_task = PythonOperator(
        task_id="data_transformations",
        python_callable=data_transformations
    )
    data_transformations_task.doc_md = dedent("Transformation of training and testing data")

    model_trainer_task = PythonOperator(
        task_id="model_trainer",
        python_callable=model_trainer
    )
    model_trainer_task.doc_md = dedent("Training of the model")

    data_ingestion_task >> data_transformations_task >> model_trainer_task
