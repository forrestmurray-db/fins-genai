# Databricks notebook source
# MAGIC %pip install -q --upgrade databricks-sdk datasets bertopic safetensors databricks-vectorsearch==0.20 databricks-genai-inference==0.1.1 mlflow==2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG forrest_murray;
# MAGIC USE SCHEMA speech;

# COMMAND ----------

audio_ds = spark.table('generated_chats_with_audio')

# COMMAND ----------

messages_list = audio_ds.collect()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # GenAI for Call Center
# MAGIC
# MAGIC ## We'll see:
# MAGIC
# MAGIC - How can customers use Databricks to process call recordings? 
# MAGIC - Use GenAI to generate insights about their calls
# MAGIC - Find the 'needles in the haystacks' by using natural language
# MAGIC - Use the call data to intelligently answer questions using RAG
# MAGIC
# MAGIC
# MAGIC ## Key platform capabilities: 
# MAGIC
# MAGIC - Unity Catalog
# MAGIC - Serverless model serving
# MAGIC - Foundation models
# MAGIC - VectorSearch

# COMMAND ----------

from utils import display_audio_chat

display_audio_chat(messages_list, with_content=False, n=5)

# COMMAND ----------

def get_b64_records(msg):
  b64_list = []
  audio_bytes = msg['audio']
  audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
  b64_list.append(audio_b64)
  return b64_list

w = WorkspaceClient()

for msg in messages_list[:5]:
  response = w.serving_endpoints.query(
      name="whisper_large_v3",
      dataframe_records=get_b64_records(msg)
  )
  print(f"{msg['role']} ({msg['voice']}) -- {response.predictions[0]}")

# COMMAND ----------

from utils import display_audio_chat

display_audio_chat(messages_list, n=5)

# COMMAND ----------

# MAGIC %md
# MAGIC # Extracting Topics

# COMMAND ----------

transcript_text = spark.table('forrest_murray.speech.generated_chats').select('content').collect()
transcript_text = [r['content'] for r in transcript_text]

# COMMAND ----------

from databricks_genai_inference import ChatCompletion
import json

chats_pandas = audio_ds.toPandas()

def mixtral_summarize_chat(group):
  messages = []
  for idx, row in group.iterrows():
    messages.append({"role": row['role'], "content": row['content']})

  prompt = """
    summarize the conversation. please respond only with the following JSON schema and no additional text:
    {"summary": <a short summary of the conversation>,
      "labels": [label 1, label 2, label 3]}
  """
  messages.append({"role": "user", "content": prompt})
  response = ChatCompletion.create(model="llama-2-70b-chat", messages=messages, max_tokens=512)
  # print(response)
  response_parsed = json.loads(response.message)
  
  group['summary'] = response_parsed["summary"]
  group['label_0'] = response_parsed["labels"][0]
  group['label_1'] = response_parsed["labels"][1]
  group['label_2'] = response_parsed["labels"][2]
  return group
  

chats_pandas = chats_pandas.groupby('topic').apply(mixtral_summarize_chat)
chats_pandas

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id, col

labeled_df = spark.table('forrest_murray.speech.customer_support_summarized_labeled')
chat_id_df = labeled_df.groupBy('topic').agg(monotonically_increasing_id().alias('chat_id'))
labeled_df_with_chat_ids = labeled_df.join(chat_id_df, on='topic')
labeled_df_with_chat_ids.write.mode('overwrite').option("overwriteSchema", "true").saveAsTable('forrest_murray.speech.customer_support_summarized_labeled')


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Retrieval Augmented Generation

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
embeddings = [e['embedding'] for e in response.data]
print(embeddings)

# COMMAND ----------

import mlflow.deployments
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql import functions as F

deploy_client = mlflow.deployments.get_deploy_client("databricks")

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    def get_embeddings(batch):
        #Note: this will gracefully fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

(spark.table('forrest_murray.speech.customer_support_summarized_labeled')
      .withColumn('summary_embedding', get_embedding('summary'))
      # .withColumn('content_embedding', get_embedding('content'))
      .drop("text", "audio")
      .write.mode('overwrite').option("overwriteSchema", "true").saveAsTable("customer_support_embedded"))

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE forrest_murray.speech.customer_support_embedded SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

catalog = "forrest_murray"
db = "speech"

vs_index_fullname = f"{catalog}.{db}.customer_support_summary_vs_index"
source_table_fullname = f"{catalog}.{db}.customer_support_embedded"

vsc.create_delta_sync_index(
  endpoint_name="shared-demo-endpoint",
  index_name=vs_index_fullname,
  source_table_name=source_table_fullname,
  pipeline_type="TRIGGERED",
  primary_key="chat_id",
  embedding_dimension=1024, #Match your model embedding size (bge)
  embedding_vector_column="summary_embedding"
)


# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

question = "Weather related incidents"
response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})
embeddings = [e['embedding'] for e in response.data]

results = vsc.get_index("shared-demo-endpoint", vs_index_fullname).similarity_search(
  query_vector=embeddings[0],
  columns=["summary", "chat_id"],
  num_results=3)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

docs_string = "\n".join([d[0] for d in docs])
docs_string

# COMMAND ----------

from databricks_genai_inference import ChatCompletion, ChatSession

agent_message = "you are a helpful assistant"

agent = ChatSession(model="llama-2-70b-chat", system_message=agent_message, max_tokens=512)
agent.reply(f"How can we improve our response to weather related incidents? Use the following documents for reference {docs_string}")

# COMMAND ----------

displayHTML(agent.last.replace("\n", "<br>"))

# COMMAND ----------


