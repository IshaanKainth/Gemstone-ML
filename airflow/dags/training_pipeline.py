from __future__ import annotations
import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from 


with DAG(
    "Gemstone_Training_Pipeline"
    default_args={"retries": 2},
    description="It is Training Pipeline",
    schedule="@weekly",
    start_date=pendulum.datetime(2025,5,15,tz="UTC")
    catchup=False
    tags=["machine_learning","regression","gemstone"]
) as dag:

dag.doc_md=__doc__

def data_ingestion(**kwargs):
    ti=kwargs["ti"]
    train_data_path,test_data_path=training_pipeline.start_data_ingestion()
    ti.xcom_push("data_ingestion_artifact",{"train_data_path";train_data_path,"test_data_path":test_data_path})

def data_transformations(**kwargs):
    ti=kwargs["ti"]
    data_ingestion_artifact=ti.xcom_pull(task_ids="data_ingestion",key="data_ingestion_artifact")    
    train_arr,test_arr=training_pipeline.start_data_transformation(data_ingestion_artifact)
    train_arr=train_arr.tolist()
    test_arr=test_arr.tolist()
    ti.xcom_push("data_transformation_artifact",{"train_arr":train_arr,"test_arr":test_arr})

def model_trainer(**kwargs):
    import numpy as np
    ti=kwargs["ti"]
    data_transformation_artifact=xcom_pull(task_ids="data_transformations",key="data_transformation_artifact")
    train_arr=np.array(data_transformation_artifact["train_arr"])
    test_arr=np.array(data_transformation_artifact["test_arr"])
    


