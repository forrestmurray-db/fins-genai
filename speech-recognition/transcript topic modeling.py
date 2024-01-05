# Databricks notebook source
# MAGIC %pip install -q -U bertopic safetensors datasets soundfile librosa

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from datasets import load_dataset

transcripts = load_dataset("google/fleurs", "en_us", split="train")

# COMMAND ----------

transcripts['raw_transcription']

# COMMAND ----------

from bertopic import BERTopic

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(transcripts['raw_transcription'])

# COMMAND ----------

topic_model.get_topic_info()

# COMMAND ----------

topic_model.visualize_topics()

# COMMAND ----------


