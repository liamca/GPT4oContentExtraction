# Azure OpenAI GPT-4o Content Extraction
Using Azure OpenAI GPT 4o to extract information such as text, tables and charts from Documents to Markdown.

There is a lot if information contained within documents such as PDF's, PPT's, and Excel Spreadsheets beyond just text, such as images, tables and charts. The goal of this repo is to show how Azure OpenAI GPT 4o can be used to extract all of this information into a Markdown file to be used for downstream processes such as RAG (Chat on your Data) or Workflows.

## Requirements

* Azure OpenAI with GPT 4o enabled
* Linux (Ubuntu) based Jupyter Notebook
* (Optional) Azure AI Search - To test the ability to answer questions
* (Optional) LibreOffice - IF you wish to support file types other than PDF

## Processing Flow
![image](https://github.com/liamca/GPT4oContentExtraction/assets/3432973/8db4eee3-6a9a-4cdd-9c7b-07ad8effd419)

## Geting Started

Ensure you have installed requirements.txt
```code
pip install -r requirements.txt
```

Install LibreOffice by executing [install libreoffice.ipynb]()


