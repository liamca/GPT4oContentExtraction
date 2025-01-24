# Dockerized Document Processor

This docker container is a fully contained service which receives an API request to process a document and do all the processing needed to make it suitable for RAG usage. It is multimodal which means that it will not only understand the text, but also the images, charts, graphs and other data within the document. Specifically this container:

- Supports files of type PPT, PPTX, XLS, XLSX, DOC, DOCX, PDF as well as web pages
- Will extract everything seen in these documents into markdown format
- Upload all of the results (markdown, images, JSON, etc) to an Azure Blob Container 
- Chunk the content is the way that is best suited for that document type. For example, PPTX will be chunked by slides, whereas PDF will be chunked by title headings
- Optionally vectorize the markdown chunks and store results into an Azure AI Search ready JSON file
- Optionally index the content into Azure AI Search

## NEW - UX for End to End Document Processing and Chat
Within the updated container, there is a UX which is accessible from the base URL of the deployed container. Using this UX, you can automatically process content, upload to Azure AI Search and perform Chat.

![image](https://github.com/user-attachments/assets/f4459995-4f02-4a77-98d1-433c10db7c8f)

## How to build the Container Image

```code
docker build -t doc2md .
```

## How to Test Container Image Locally

```code
docker run -d -p 3100:3100 gpt4odemo
```
Validate it is running:
```code
docker ps
```
Using your IP address, open browser and go to: X.X.X.X:3100

## How to upload the container to Azure
Create an Azure Container Registry 

Login to Azure and then login to Registry
```code
az login
az acr login -n <registry>
```

Tag the container:
```code
docker tag doc2md <registry>.azurecr.io/insight_engine/doc2md:v1
```

Upload the container to registry:
```code
docker push <registry>.azurecr.io/insight_engine/doc2md:v1
```

## How to test the container

IMPORTANT:
- This container can not process Office documents with Sensitivity levels that are not public / general as it will not be able to access the content.
- You can optionally pass a parameter "chunk_type" to override the chunking that is done. Valid value are "page" and "markdown".
- The is_html flag in the below parameters should be set to False unless the URL you will be providing is an HTML page:

After you create the Azure Container App use the following example to test the container.
[test-process-document.ipynb](https://github.com/liamca/GPT4oContentExtraction/blob/main/docker/test-process-document.ipynb)

Within this notebook, you will need to make changes to the data that is submitted by replacing <redacted> with your service details.


```code
data = {  
    "prompt": """Extract everything you see in this image to markdown. 
                Convert all charts such as line, pie and bar charts to markdown tables and include a note that the numbers are approximate.
                """,
    "openai_gpt_api_base" : "https://[redacted].openai.azure.com/",
    "openai_gpt_api_key" : "[redacted]",
    "openai_gpt_api_version" :  "2024-02-15-preview",
    "openai_gpt_model" : "gpt-4o",
    "blob_storage_service_name" : "[redacted]",
    "blob_storage_service_api_key" : "[redacted]",
    "blob_storage_container" : "doc2md",
    "openai_embedding_api_base" : "https://[redacted].openai.azure.com/",
    "openai_embedding_api_key" : "[redacted]",
    "openai_embedding_api_version" :  "2024-02-15-preview",
    "openai_embedding_model" : "text-embedding-ada-002",
    "search_service_name": "[redacted]",
    "search_admin_key" : "[redacted]",
    "search_index_name": "[redacted]",
    "search_api_version" : "2024-05-01-preview"
}  
```

Set this to be the URL  location of the file you want to process
```code
data['url_file_to_process'] = "https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search"
```

Set this to be the Azure Container App that you created
```code
base_url = "https://<redacted>.westus2.azurecontainerapps.io"
```
