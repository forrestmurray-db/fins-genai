# Databricks notebook source
# MAGIC %pip install -q langchain langchain-experimental "unstructured[all-docs]" openai chroma lancedb eparse

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Using Excel Document Loaders
# MAGIC
# MAGIC We can use `unstructured` document loaders to extract, clean, format, etc. a variety of different document types. 
# MAGIC We will start by loading the document, then split it into embeddable sized pieces, and save it in a vector store. Then we will try retrieving document chunks based on a relevant query. 

# COMMAND ----------

from langchain_community.document_loaders import UnstructuredExcelLoader

# COMMAND ----------

loader = UnstructuredExcelLoader("./eparse_unit_test_data.xlsx", mode="elements")
docs = loader.load()
docs[0]

# COMMAND ----------

displayHTML(docs[0].metadata['text_as_html'])

# COMMAND ----------

import os
import openai

openai.api_key = dbutils.secrets.get("access-tokens", 'openai')
os.environ["OPENAI_API_KEY"] = openai.api_key

# COMMAND ----------

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain.text_splitter import CharacterTextSplitter

embeddings = OpenAIEmbeddings()
documents = CharacterTextSplitter().split_documents(docs)

# COMMAND ----------

import lancedb

db = lancedb.connect("/tmp/lancedb")
table = db.create_table(
    "my_table",
    data=[
        {
            "vector": embeddings.embed_query("Hello World"),
            "text": "Hello World",
            "id": "1",
        }
    ],
    mode="overwrite",
)

docsearch = LanceDB.from_documents(documents, embeddings, connection=table)

# COMMAND ----------

retrieved_chunks = docsearch.similarity_search('ABC Company EBITDA')

retrieved_chunks

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC From the above results we can see that we are able to retrieve some relevant results in our vector store related to the query of 'ABC Company EBITDA'. This is a very basic example. Let's try to do something a little more sophisticated. 
# MAGIC
# MAGIC # 

# COMMAND ----------

from functools import partial

from langchain.chains.combine_documents import collapse_docs, split_list_of_docs
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

llm = ChatOpenAI(model='gpt-4-1106-preview')

# COMMAND ----------

document_prompt = PromptTemplate.from_template("{page_content}")
partial_format_document = partial(format_document, prompt=document_prompt)

# COMMAND ----------

# map_chain = (
#     {"context": partial_format_document}
#     | PromptTemplate.from_template("Summarize this content:\n\n{context}")
#     | llm
#     | StrOutputParser()
# )

from langchain.chains import create_extraction_chain
from langchain_community.chat_models import ChatOpenAI

schema = {
    "properties": {
        "spreadsheet_topic": {"type": "string"},
        "spreadsheet_entity": {"type": "integer"},
        "spreadsheet_fields": {"type": "string"},
    },
    "required": ["spreadsheet_topic", "spreadsheet_fields"],
}

map_chain = create_extraction_chain(schema, llm)

# COMMAND ----------

from langchain.schema import Document

map_as_doc_chain = (
    RunnableParallel({"doc": RunnablePassthrough(), "content": map_chain})
    | (lambda x: Document(page_content=x["content"], metadata=x["doc"].metadata))
).with_config(run_name="Summarize (return doc)")

# COMMAND ----------

def format_docs(docs):
    return "\n\n".join(partial_format_document(doc) for doc in docs)


collapse_chain = (
    {"context": format_docs}
    | PromptTemplate.from_template("Collapse this content:\n\n{context}")
    | llm
    | StrOutputParser()
)


def get_num_tokens(docs):
    return llm.get_num_tokens(format_docs(docs))


def collapse(
    docs,
    config,
    token_max=4000,
):
    collapse_ct = 1
    while get_num_tokens(docs) > token_max:
        config["run_name"] = f"Collapse {collapse_ct}"
        invoke = partial(collapse_chain.invoke, config=config)
        split_docs = split_list_of_docs(docs, get_num_tokens, token_max)
        docs = [collapse_docs(_docs, invoke) for _docs in split_docs]
        collapse_ct += 1
    return docs

# COMMAND ----------

reduce_chain = (
    {"context": format_docs}
    | PromptTemplate.from_template("Combine these summaries:\n\n{context}")
    | llm
    | StrOutputParser()
).with_config(run_name="Reduce")

# COMMAND ----------

map_reduce = (map_as_doc_chain.map() | collapse | reduce_chain).with_config(
    run_name="Map reduce"
)

# COMMAND ----------

print(map_reduce.invoke(docs, config={"max_concurrency": 5}))

# COMMAND ----------

from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# COMMAND ----------

agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-4-1106-preview"),
    "titanic.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

# COMMAND ----------

from langchain.globals import set_debug

set_debug(True)

agent.run("how many rows are there?")

# COMMAND ----------


