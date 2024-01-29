# Databricks notebook source
!pip install -U langchain openai chromadb langchain-experimental typing_extensions sqlalchemy "urllib3>=1,<2"

# COMMAND ----------

!pip install -U "unstructured[pdf]" "unstructured[local-inference]" pillow pydantic lxml pillow matplotlib tiktoken open_clip_torch torch

# COMMAND ----------

!apt-get install -y poppler-utils tesseract-ocr libmagic-dev

# COMMAND ----------

!pip install "urllib3>=1,<2"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from typing import Any
import mlflow
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

path = "/Volumes/forrest_murray/docvqa/arxiv_papers/1706.03762v7.pdf"
output_path = "/Volumes/forrest_murray/docvqa/arxiv_papers/images/"

# update path and output path to a UC volume or DBFS path containing a source document

raw_pdf_elements = partition_pdf(
    filename=path,                  # mandatory
    strategy="hi_res",                                     # hi res uses https://github.com/facebookresearch/detectron2
    extract_images_in_pdf=True,                            # mandatory to set as ``True``
    extract_image_block_types=["Image", "Table"],          # block extract will extract table as jpg
    extract_image_block_to_payload=False,                  # optional
    extract_image_block_output_dir=output_path,  # optional - only works when ``extract_image_block_to_payload=False``
    url=None
)

# COMMAND ----------

raw_pdf_elements

# COMMAND ----------

from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.basic import chunk_elements

chunks = chunk_by_title(raw_pdf_elements,
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,)

# COMMAND ----------

from uuid import uuid4
import base64
from openai import OpenAI

# Add your openai api key here
client = OpenAI(api_key=dbutils.secrets.get('access-tokens', 'openai'))


class MMImage:

  def __init__(self, img):
    self._type = str(type(img))
    self._element = img
    self.id = str(uuid4())
    self.image_path = self._element.metadata.image_path
    self.caption = None
    self.summary = None

  def encode(self):
    with open(self.image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')
    
  def gpt_summarize(self, model=None, prompt=None):
    base64_image = self.encode()

    prompt = prompt or "Describe the image in detail. Be specific about graphs, such as bar plots."

    if self.caption:
      prompt = prompt + f"A caption for the image {self.caption}"

    response = client.chat.completions.create(
      model="gpt-4-vision-preview",
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": prompt},
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

    self.summary = response.choices[0].message.content
    return self.summary

  def to_dict(self):
    return dict(
      summary=self.summary,
      id=self.id,
      image_path=self.image_path,
      caption=self.caption,
      _type=self._type
    )
  
  
class MMTable:

  def __init__(self, table):
    self._type = str(type(table))
    self._element = table
    self.id = str(uuid4())
    self.image_path = self._element.metadata.image_path
    self.caption = None
    self.summary = None
    self.text = str(table)

  def encode(self):
    with open(self.image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

  def gpt_summarize(self, prompt=None):
    base64_image = self.encode()

    prompt = prompt or f"Describe the table in detail. An extracted text version of the table: {self.text}"

    if self.caption:
      prompt = prompt + f"A possible caption for the table: {self.caption}"

    response = client.chat.completions.create(
      model="gpt-4-vision-preview",
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": prompt},
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

    self.summary = response.choices[0].message.content
    return self.summary
  
  def to_dict(self):
    return dict(
      summary=self.summary,
      id=self.id,
      image_path=self.image_path,
      caption=self.caption,
      text=self.text,
      _type=self._type
    )
  

# COMMAND ----------

import pandas as pd

def group_elements(raw_elements):
  images = []
  tables = []
  texts = []

  for el, next_el in zip(raw_elements, raw_elements[1:]):
    if "unstructured.documents.elements.Image" in str(type(el)):
      img = MMImage(el)
      if  "unstructured.documents.elements.FigureCaption" in str(type(next_el)):
        img.caption = str(next_el)
      images.append(img)
    elif "unstructured.documents.elements.Table" in str(type(el)):
      tbl = MMTable(el)
      tbl.caption = next_el
      tables.append(tbl)
    else: 
      texts.append(el)

  return dict(
    images=images,
    tables=tables,
    texts=texts
  )

grouped_elements = group_elements(raw_pdf_elements)
images, tables, _ = grouped_elements.values()
texts = [c for c in chunks if "unstructured.documents.elements.CompositeElement" in str(type(c))]

# COMMAND ----------

import concurrent.futures

non_texts = images + tables

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(el.gpt_summarize) for el in non_texts]

    summaries = [future.result() for future in futures]

# COMMAND ----------

from PIL import Image
import io

def display_mm_image(img):
  with open(img.image_path, "rb") as f:
    display(Image.open(io.BytesIO(f.read())))
    display(img.summary)

display_mm_image(images[0])

# COMMAND ----------

df = pd.DataFrame([instance.to_dict() for instance in images + tables])

# Set all columns to string type
df = df.astype(str)
delta_table_full_name = "forrest_murray.docvqa.extracted_elements" # delta table for persisting elements and summaries, add yours here
els_df = spark.createDataFrame(df).write.mode('overwrite').saveAsTable(delta_table_full_name)

# COMMAND ----------

import os
import chromadb
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.schema import Document
from PIL import Image as _PILImage

id_key = "doc_id"

# Create chroma
vectorstore = Chroma(
    collection_name="clip_5", embedding_function=OpenCLIPEmbeddings()
)

# Add images
vectorstore.add_images(uris=[im.image_path for im in non_texts], metadatas=[{id_key: i.id, "doc_type": "source image"} for i in non_texts])

# Make retriever
retriever = vectorstore.as_retriever()

# COMMAND ----------

def display_retrieval_results(queries, k=1):
  for query in queries:
    print("QUERY: " + query)
    for result in retriever.get_relevant_documents(query)[:k]:
      print("Document Type: " + result.metadata["doc_type"])
      if result.page_content.endswith('.jpg'):
        with open(result.page_content, "rb") as f:
          display(Image.open(io.BytesIO(f.read())))
      else: 
        displayHTML(result.page_content.replace('\n', '<br>'))
      print(" --------------------- ")

# COMMAND ----------

display_retrieval_results(["model architecture", "memory", "multi-hop"], k=1)

# COMMAND ----------

# The storage layer for the parent documents
store = InMemoryStore()

summary_docs = [
    Document(page_content=i.summary, metadata={id_key: i.id, "doc_type": "gpt summary"})
    for i in images + tables
]

doc_ids = [ j.id for j in images + tables ]

text_ids = [str(uuid4()) for _ in texts]
text_docs = [
    Document(page_content=str(text), metadata={id_key: text_ids[i], "doc_type": "source text"})
    for i, text in enumerate(texts)
]

# Add documents
vectorstore.add_documents(summary_docs + text_docs)

# Add to kv-store
store.mset(list(zip(doc_ids, [d for d in images + tables + texts])))

# COMMAND ----------

display_retrieval_results(["multi-hop"], k=4)

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
        if doc.endswith('.jpg'):
            with open(doc, "rb") as f:
                doc = base64.b64encode(f.read()).decode('utf-8')

        if is_base64(doc):
            # Resize image to avoid OAI server error
            images.append(
                resize_base64_image(doc, size=(250, 250))
            )  # base64 encoded str
        else:
            text.append(doc)
    return {"images": images, "texts": text}

# COMMAND ----------

from operator import itemgetter

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI


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
    # text_message = {
    #     "type": "text",
    #     "text": (
    #         "As a machine learning subject matter expert, your task is to analyze and interpret figures from published research papers, "
    #         "considering their implications and significance. Alongside the images, you will be "
    #         "provided with related text to offer context. Both will be retrieved from a vectorstore based "
    #         "on user-input keywords. Please use your extensive knowledge and analytical skills to provide a "
    #         "comprehensive summary that includes:\n"
    #         "- A detailed description of the visual elements in the image.\n"
    #         "- The significance of the image's contribution to the field of machine learning.\n"
    #         "- Connections between the image and the related text.\n\n"
    #         f"User-provided keywords: {data_dict['question']}\n\n"
    #         "Text and / or tables:\n"
    #         f"{formatted_texts}"
    #     ),
    # }

    text_message = {
        "type": "text",
        "text": (
            "As a machine learning subject matter expert, your task is to analyze and interpret figures from published research papers, "
            "considering their implications and significance. Alongside the images, you will be "
            "provided with related text to offer context. Both will be retrieved from a vectorstore based "
            "on user-input keywords. Please use your extensive knowledge and analytical skills to provide an answer to the question "
            f"Question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    
    messages.append(text_message)

    return [HumanMessage(content=messages)]


model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024, openai_api_key=dbutils.secrets.get("access-tokens", 'openai'))

# RAG pipeline
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

from langchain.globals import set_debug

# uncomment to see the full langchain logs
# set_debug(True)

# COMMAND ----------

chain.invoke("What was the best performing model on the Penn Treebank corpora?")
