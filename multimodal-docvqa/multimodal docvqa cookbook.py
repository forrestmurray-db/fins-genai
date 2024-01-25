# Databricks notebook source
!pip install -U langchain unstructured[all-docs] pydantic lxml mlflow>=2.9

# COMMAND ----------

!pip install "urllib3>=1,<2"

# COMMAND ----------

!apt-get install -y poppler-utils tesseract-ocr libmagic-dev

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from typing import Any

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

# path = "/Volumes/forrest_murray/docvqa/arxiv_papers/"

path = "/Volumes/forrest_murray/docvqa/arxiv_papers/1503.08895v5.pdf"

# Get elements
raw_pdf_elements = partition_pdf(
    filename=path,
    # Using pdf format to find embedded image blocks
    extract_images_in_pdf=True,
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any sub-section of the document
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    # Hard max on chunks
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,
)

# COMMAND ----------

# Create a dictionary to store counts of each type
category_counts = {}

for element in raw_pdf_elements:
    category = str(type(element))
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1

# Unique_categories will have unique elements
unique_categories = set(category_counts.keys())
category_counts

# COMMAND ----------

class Element(BaseModel):
    type: str
    text: Any


# Categorize by type
categorized_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))

# Tables
table_elements = [e for e in categorized_elements if e.type == "table"]
print(len(table_elements))

# Text
text_elements = [e for e in categorized_elements if e.type == "text"]
print(len(text_elements))

# COMMAND ----------

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks

# Prompt
prompt_text = """You are an assistant tasked with summarizing tables and text. \
Give a concise summary of the table or text. Table or text chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
# model = Databricks(endpoint_name="databricks-llama-2-70b-chat")
model = ChatDatabricks(target_uri="databricks", endpoint="databricks-mixtral-8x7b-instruct")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# COMMAND ----------

# Apply to text
texts = [i.text for i in text_elements]
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

# COMMAND ----------

# Apply to tables
tables = [i.text for i in table_elements]
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

# COMMAND ----------

import json
# Save the tables and table summaries to a file
with open("./data.json", "w") as f:
  dict(
    text_elements = text_elements,
    texts = texts,
    text_summaries = text_summaries,
    tables = tables,
    table_summaries = table_summaries,
    table_elements = table_elements
  )



# COMMAND ----------

!pip install -U openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from langchain_community.llms import OpenAI
from openai import OpenAI
import os
import base64

img_summaries = []

client = OpenAI(api_key=dbutils.secrets.get("access-tokens", 'openai'))

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

for i in os.listdir('./figures'):
  base64_image = encode_image(f'./figures/{i}')

  response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe the image in detail. Be specific about graphs, such as bar plots."},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ],
      }
    ],
    max_tokens=300,
  )

  img_summaries.append(response.choices[0])

# vllm = OpenAI(openai_api_key=dbutils.secrets.get("access-tokens", 'openai'))

# COMMAND ----------

img_summaries = [i.message.content for i in img_summaries]

# COMMAND ----------

# MAGIC %pip install chromadb

# COMMAND ----------



# COMMAND ----------

import json
# Save the tables and table summaries to a file
with open("./data.json", "w") as f:
  data = json.dump(dict(
    # text_elements = text_elements,
    texts = texts,
    text_summaries = text_summaries,
    tables = tables,
    table_summaries = table_summaries,
    # table_elements = table_elements,
    img_summaries = img_summaries
  ), f)


# COMMAND ----------

import json

with open("data.json", "rb") as f:
  data = json.load(f)
texts, text_summaries, tables, table_summaries = data.values()

# COMMAND ----------

import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import DatabricksEmbeddings

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries", embedding_function=DatabricksEmbeddings(endpoint="databricks-bge-large-en"))

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))

# COMMAND ----------

# path = "/Volumes/forrest_murray/docvqa/arxiv_papers/"
figpath = './figures/'

image_uris = sorted(
    [
        os.path.join(figpath, image_name)
        for image_name in os.listdir(figpath)
        if image_name.endswith(".jpg")
    ]
)

retriever.vectorstore.add_images(uris=image_uris)

# COMMAND ----------

cleaned_img_summary = img_summaries

img_ids = [str(uuid.uuid4()) for _ in cleaned_img_summary]
summary_img = [
    Document(page_content=s, metadata={id_key: img_ids[i]})
    for i, s in enumerate(cleaned_img_summary)
]

retriever.vectorstore.add_documents(summary_img)
### Fetch images
retriever.docstore.mset(
    list(
        zip(
            img_ids,
            summary_img
        )
    )
)

# COMMAND ----------

tables[3]

# COMMAND ----------

table_summaries[3]

# COMMAND ----------

retriever.get_relevant_documents(
    "What was perplexity on the test sets of Penn Treebank and Text8 corpora?"
)[1]

# COMMAND ----------

import base64
import io
from io import BytesIO

import numpy as np
from PIL import Image


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string.

    Args:
    base64_string (str): Base64 string of the original image.
    size (tuple): Desired size of the image as (width, height).

    Returns:
    str: Base64 string of the resized image.
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False


def split_image_text_types(docs):
    """Split numpy array images and texts"""
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content  # Extract Document contents
        if is_base64(doc):
            # Resize image to avoid OAI server error
            images.append(
                resize_base64_image(doc, size=(250, 250))
            )  # base64 encoded str
        else:
            text.append(doc)
    return {"images": images, "texts": text}

# COMMAND ----------

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI

def prompt_func(data_dict):
    # Joining the context texts into a single string
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
            },
        }
        messages.append(image_message)

    # Adding the text message for analysis
    text_message = {
        "type": "text",
        "text": (
            "As a machine learning subject matter expert, your task is to analyze and interpret figures from published research papers, "
            "considering their implications and significance. Alongside the images, you will be "
            "provided with related text to offer context. Both will be retrieved from a vectorstore based "
            "on user-input keywords. Please use your extensive knowledge and analytical skills to provide a "
            "comprehensive summary that includes:\n"
            "- A detailed description of the visual elements in the image.\n"
            "- The significance of the image's contribution to the field of machine learning.\n"
            "- Connections between the image and the related text.\n\n"
            f"User-provided keywords: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)

    return [HumanMessage(content=messages)]


model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024, openai_api_key=dbutils.secrets.get("access-tokens", 'openai'))

chain = (
    {
        "context": retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(prompt_func)
    | model
    | StrOutputParser()
)

# COMMAND ----------

from langchain_core.runnables import RunnablePassthrough

# Prompt template
template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Option 1: LLM
# model = ChatOpenAI(temperature=0, model="gpt-4")
# Option 2: Multi-modal LLM
# model = GPT4-V or LLaVA

# RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
