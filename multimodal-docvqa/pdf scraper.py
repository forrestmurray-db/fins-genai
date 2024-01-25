# Databricks notebook source
# MAGIC %sql
# MAGIC USE catalog forrest_murray;
# MAGIC CREATE SCHEMA docvqa;
# MAGIC USE SCHEMA docvqa;
# MAGIC
# MAGIC CREATE VOLUME arxiv_papers;

# COMMAND ----------

volume_path = '/Volumes/forrest_murray/docvqa/arxiv_papers/'

import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm

# Base URL of the website
base_url = 'https://paperswithcode.com'

# Task-specific URL
task_url = '/task/question-answering#papers-list'

# Function to get soup object
def get_soup(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BeautifulSoup(response.text, 'html.parser')
    else:
        print("Failed to retrieve the webpage")
        return None

# Function to download PDF
def download_pdf(pdf_url, save_dir):
    response = requests.get(pdf_url)
    pdf_name = pdf_url.split('/')[-1]
    with open(os.path.join(save_dir, pdf_name), 'wb') as f:
        f.write(response.content)
    print(f"Downloaded: {pdf_name}")

# Function to scrape and download PDFs
def scrape_and_download_pdfs(base_url, task_url):
    soup = get_soup(base_url + task_url)
    if soup:
        paper_links = soup.find_all('a', href=True, class_="badge badge-light")
        paper_urls = [base_url + link['href'] for link in paper_links if link['href'].startswith('/paper/')]

        # Directory to save PDFs
        save_dir = volume_path
        os.makedirs(save_dir, exist_ok=True)

        for paper_url in tqdm(paper_urls):
            paper_soup = get_soup(paper_url)
            if paper_soup:
                pdf_link = paper_soup.find('a', href=True, class_="badge badge-light", onclick=True)
                if pdf_link and 'arxiv.org' in pdf_link['href']:
                    download_pdf(pdf_link['href'], save_dir)

# Start the scraping and downloading process
scrape_and_download_pdfs(base_url, task_url)


# COMMAND ----------


