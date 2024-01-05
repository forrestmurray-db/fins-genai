# Databricks notebook source
# MAGIC %pip install --upgrade databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from datasets import load_dataset

audio_ds = load_dataset("Nexdata/accented_english", split="train")

# COMMAND ----------

from datasets import load_dataset
import pandas as pd
import base64
import json

from databricks.sdk import WorkspaceClient

sample_path = dataset[0]["audio"]["path"]

with open(sample_path, 'rb') as audio_file:
    audio_bytes = audio_file.read()
    audio_b64 = base64.b64encode(audio_bytes).decode('ascii')

dataframe_records = [audio_b64]

w = WorkspaceClient()
response = w.serving_endpoints.query(
    name="whisper-v3l",
    dataframe_records=dataframe_records,
)
print(response.predictions)

# COMMAND ----------

from IPython.display import Audio
import io

with open(sample_path, 'rb') as audio_file:
  audio_bytes = audio_file.read()
  audio_io = io.BytesIO(audio_bytes)
  audio = Audio(audio_io.read())

  display(audio)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Batch Inference

# COMMAND ----------

fleurs = load_dataset("google/fleurs", "en_us", split="train")

# COMMAND ----------

fleurs_path = '/dbfs/fleurs/en/en.parquet'
fleurs.to_parquet(fleurs_path)

# COMMAND ----------

fleurs_df = spark.read.parquet('/fleurs/en')

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG forrest_murray;
# MAGIC CREATE SCHEMA IF NOT EXISTS speech;
# MAGIC USE SCHEMA speech;

# COMMAND ----------

fleurs_df.count()

# COMMAND ----------

fleurs_df.write.saveAsTable('fleurs_en')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC ai_query(
# MAGIC   'databricks_whisper_v3_model_forrest.models.whisper_large_v3',
# MAGIC   audio.bytes,
# MAGIC   'returnType',
# MAGIC   'STRING'
# MAGIC ) as transcription
# MAGIC from fleurs_en

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

catalog_name = "databricks_whisper_v3_models"
transcribe = mlflow.pyfunc.spark_udf(spark, f"models:/{catalog_name}.models.whisper_large_v3/1", "string")
