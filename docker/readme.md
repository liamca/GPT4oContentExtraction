# Dockerized Document Processor

This docker container is a fully contained service which receives an API request to process a document and do all the processing needed to make it suitable for RAG usage. It is multimodal which means that it will not only understand the text, but also the images, charts, graphs and other data within the document. Specifically this container:

- Supports files of type PPT, PPTX, XLS, XLSX, DOC, DOCX, PDF as well as web pages
- Will extract everything seen in these documents into markdown format
- Chunk the content is the way that is best suited for that document type. For example, PPTX will be chunked by slides, whereas PDF will be chunked by title headings
- Optionally vectorize the markdown chunks and store results into an Azure AI Search ready JSON file
- Optionally index the content into Azure AI Search

## How to build the Container

```code
docker build -t doc2md .
```

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

After this, you can create an Azure Container App which leverages this container and use the following example to test the container.
[test-process-document.ipynb](https://github.com/liamca/GPT4oContentExtraction/blob/main/docker/test-process-document.ipynb)

