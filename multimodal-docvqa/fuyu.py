# Databricks notebook source
# MAGIC %sh
# MAGIC cd .. && ls -sail

# COMMAND ----------

# MAGIC %run ./_includes/_setup

# COMMAND ----------

# MAGIC %pip install -U transformers

# COMMAND ----------

# MAGIC %sql 
# MAGIC USE CATALOG forrest_murray;
# MAGIC USE SCHEMA models;
# MAGIC -- CREATE VOLUME forrest_murray.models.fuyu;

# COMMAND ----------

import os 

os.environ["HF_CACHE"] = "/Volumes/forrest_murray/models/fuyu"
os.environ["TRANSFORMERS_CACHE"] = ""

# COMMAND ----------

import torch
torch.cuda.get_device_properties(0)

# COMMAND ----------

!pip install urllib3==1.25.11

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests
import torch

model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

# COMMAND ----------

model.save_pretrained("/Volumes/forrest_murray/models/fuyu/")

# COMMAND ----------

# prepare inputs for the model
# text_prompt = "Generate a coco-style caption.\n"
# text_prompt = "What instrument is the toy bear playing?\n"
# url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
# url = "https://www.adept.ai/images/blog/fuyu-8b/snare_bear.png"
url = "https://www.adept.ai/images/blog/fuyu-8b/red_tree_vole.png"
text_prompt = "If in the food web shown in the diagram, Douglas fir tree needles are absent, which organism would starve?"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:0")

# autoregressively generate text
generation_output = model.generate(**inputs, max_new_tokens=7)
generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
# assert generation_text == ['A blue bus parked on the side of a road.']
generation_text

# COMMAND ----------

from typing import List

def fuyu_inference(text_prompt, image) -> List[str]:
  # image = Image.open(image)
  inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:0")
  generation_output = model.generate(**inputs, max_new_tokens=20)
  generation_text = processor.batch_decode(generation_output[:, -20:], skip_special_tokens=True)
  return generation_text

# COMMAND ----------

inputs

# COMMAND ----------

# MAGIC %pip install "unstructured[all-docs]"

# COMMAND ----------

!apt-get install -y poppler-utils tesseract-ocr libmagic-dev

# COMMAND ----------

from unstructured.partition.pdf import partition_pdf
# import io

# paper = requests.get("https://arxiv.org/pdf/1706.03762.pdf")

with open('test.pdf', 'wb') as file:
  paper = requests.get("https://arxiv.org/pdf/1706.03762.pdf")
  file.write(paper.content)

doc = partition_pdf('test.pdf', strategy='hi_res')
doc

# COMMAND ----------

image_caption_pairs = [ (i,j) for i, j in zip(doc, doc[1:]) if i.category == 'Image' and j.category == 'FigureCaption']

# COMMAND ----------

image_caption_pairs[0][0].metadata.coordinates.to_dict()

# COMMAND ----------

import pdf2image

document = pdf2image.convert_from_path('test.pdf')

# COMMAND ----------

display(document[5])

# COMMAND ----------

fuyu_inference("what is the complexity per layer of Self-Attention?\n", document[5])

# COMMAND ----------

document[2].save('pdfs/3.png', 'PNG')

# COMMAND ----------

# im = Image.frombytes('RGB', document[2].size, document[2].tobytes())
im = Image.open('pdfs/3.png')

# COMMAND ----------

im

# COMMAND ----------

im = Image.open(requests.get('https://production-media.paperswithcode.com/social-images/oVEwwksZyfDziYzq.png', stream=True).raw).convert('RGB')

# COMMAND ----------

im

# COMMAND ----------

inputs = processor(text="what is pink?\n", images=im, return_tensors="pt")

for k, v in inputs.items():
    # inputs[k] = v.to("cuda:0")
    print(k, v)

# generation_output = model.generate(**inputs, max_new_tokens=10)
# generation_text = processor.batch_decode(generation_output[:, -10:], skip_special_tokens=True)

# COMMAND ----------

generation_text

# COMMAND ----------


