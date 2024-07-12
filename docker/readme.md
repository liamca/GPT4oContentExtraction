# Azure Container App 

Helpful commands to create, upload and publish the container. This assumes you already have an Azure Container Registry

## Build Container

```code
docker build -t doc2md .
```
## Tag Container
```code
docker tag doc2md [registry].azurecr.io/insight_engine/doc2md
```
## Upload Container to Azure Container Registry
```code
docker push [registry].azurecr.io/insight_engine/doc2md
```
## Upload Container to Azure Container Registry
```code
docker push [registry].azurecr.io/insight_engine/doc2md
```
## Create Azure Container App
```code
az containerapp create --resource-group [resourcegroup] --name [name] --ingress external --query properties.configuration.ingress.fqdn  --target-port 3100 --image [registry].azurecr.io/insight_engine/doc2md --environment [environment name] --cpu 2 --memory 4Gi --registry-username [registry username] --registry-password [registry password] --registry-server [registry].azurecr.io  --min-replicas 0 --max-replicas 5
```                       

