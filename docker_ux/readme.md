# UX for Performing GPT 4o Based Content Extraction and Chat

This UX works with the docker deployment found [here](https://github.com/liamca/GPT4oContentExtraction/tree/main/docker).

## Requirements:
- Docker installed and running on local machine
- Azure (AZ cli)[https://learn.microsoft.com/cli/azure/install-azure-cli]
- Azure Container registry created and logged in (az acr login -n <your-azure-container-registry>)

## Building and Upload 
```
docker build -t doc2md_ux .
docker tag doc2md_ux <your-azure-container-registry>.azurecr.io/insight_engine/doc2md_ux:v1
docker push <your-azure-container-registry>.azurecr.io/insight_engine/doc2md_ux:v1
```

