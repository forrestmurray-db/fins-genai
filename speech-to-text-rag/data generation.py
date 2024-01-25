# Databricks notebook source
!pip install databricks-genai-inference elevenlabs

dbutils.library.restartPython()

# COMMAND ----------

from databricks_genai_inference import ChatCompletion, ChatSession

agent_message = "you are a helpful assistant"

agent = ChatSession(model="mixtral-8x7b-instruct", system_message=agent_message, max_tokens=512)
agent.reply("Come up with a list of 10 things a customer would call their insurance company about, in first person POV. For example: 'My car was damaged by hail.'")

# COMMAND ----------

# customer_call_topics = [
#   "My car was damaged by hail.", 
#   "I want to add a vehicle to my policy.",
#   "I had a pipe burst in my house.",
#   "I have flood damange to my home.",
#   "Why did my premium go up?"
# ]
customer_call_topics = agent.last.split("\n")
customer_call_topics

# COMMAND ----------

from databricks_genai_inference import ChatCompletion, ChatSession
from tqdm import tqdm

chats = []

for topic in tqdm(customer_call_topics):
  agent_message = "You are a customer support agent for an insurance company. Do not ask for information about the vehicle or policy number, assume you have that information already (make up a vehicle and policy number)"
  customer_message = f"You are roleplaying a customer of an insurance company and you are chatting with an agent to get help with the following problem: {topic}"

  agent = ChatSession(model="mixtral-8x7b-instruct", system_message=agent_message, max_tokens=512)
  customer = ChatSession(model="mixtral-8x7b-instruct", system_message=customer_message, max_tokens=512)

  for i in tqdm(range(5)):
    if i % 2 == 0:
      customer_message = customer.last or topic
      agent.reply(customer_message)
    else: 
      customer.reply(agent.last)

  chats.append(agent.history)

# COMMAND ----------

import pandas as pd
data_list = []

for topic, chat in zip(customer_call_topics, chats):
  
  for msg in chat:
    data_list.append([topic, msg['role'], msg['content']])
  
cols = ["topic", "role", "content"]
df = pd.DataFrame(data_list, columns=cols)
df

# COMMAND ----------

df[['content', 'topic']] = df[['content', 'topic']].replace({'\n': ' ', '\t': ' ', r'\b[0-9]+[.:]': ' '}, regex=True)

# COMMAND ----------

sparkdf = spark.createDataFrame(df).write.mode('append').saveAsTable('generated_chats')
# chats_df = spark.read.table('forrest_murray.speech.generated_chats').toPandas()

# COMMAND ----------

from elevenlabs import generate, play, voices

# test_msg = df.iloc[1]['content']

# audio = generate(
#   text=test_msg,
#   voice="Charlie",
#   model="eleven_multilingual_v2"
# )

# COMMAND ----------

import random
from elevenlabs import set_api_key

set_api_key("<elevenlabs api token>")

agent_voice, customer_voice = random.choices(voices(), k=2)

chat_messages = chats_df[chats_df.role != 'system']

# COMMAND ----------

def assign_voices(group):
    roles = group['role'].unique()
    selected_voices = random.sample(list(voices()), len(roles))
    voice_map = dict(zip(roles, selected_voices))
    group['voice'] = group['role'].map(voice_map)
    return group
  
chat_messages = chat_messages.groupby('topic').apply(assign_voices)
chat_messages

# COMMAND ----------

chat_messages['content'] = chat_messages['content'].str.replace(r'\b[0-9]+.', '', regex=True)
chat_messages

# COMMAND ----------

chat_messages['content'] = chat_messages['content'].replace({'\n': ' ', '\t': ' ', r'\b[0-9]+[.:]': ' '}, regex=True)
chat_messages

# COMMAND ----------

def generate_audio(row):
  return generate(
    text=row['content'],
    voice=row['voice'],
    model="eleven_multilingual_v2"
  )

chat_messages['audio'] = chat_messages.apply(generate_audio, axis=1)


# COMMAND ----------

chat_messages['voice'] = chat_messages['voice'].apply(lambda x: x.name)

# COMMAND ----------

chat_messages_write = spark.createDataFrame(chat_messages).write.saveAsTable('forrest_murray.speech.generated_chats_with_audio')

# COMMAND ----------

chat_messages

# COMMAND ----------


