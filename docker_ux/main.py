import httpx  
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException   
from fastapi.responses import HTMLResponse, JSONResponse 
from fastapi.templating import Jinja2Templates  
from pydantic import BaseModel  
from typing import List, Dict  
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions  
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError 
from datetime import datetime, timedelta  
import json  
from openai import AzureOpenAI, OpenAIError
import openai
import requests
import time

app = FastAPI()  

class UploadFiles(BaseModel):  
    blob_storage_service_name: str  
    blob_storage_service_api_key: str
    blob_storage_container: str
    folder: str
  

templates = Jinja2Templates(directory="templates")  

max_tokens = 4096
openai_temperature = 0.1
  
def construct_connection_string(account_name: str, account_key: str) -> str:  
    return f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"  

def parse_doc_id(input_string):  
    # Split the string by dashes  
    parts = input_string.rsplit('-', 2)  
      
    if len(parts) != 3:  
        raise ValueError("Input string does not have the expected format with two dashes.")  
      
    base_url_encoded = parts[0]  
    chunk_id = parts[1]  
    pg_number = parts[2]  
      
    # Decode the base64-encoded base_url  
    #base_url = base64.b64decode(base_url_encoded).decode('utf-8')  
      
    return base_url_encoded, chunk_id, pg_number  


@app.post("/upload_files/")  
async def upload_files(  
    blob_storage_service_name: str = Form(...),  
    blob_storage_service_api_key: str = Form(...),  
    blob_storage_container: str = Form(...),  
    folder: str = Form(None),  
    files: List[UploadFile] = File(...)  
) -> Dict[str, List[str]]: 

    connection_string = construct_connection_string(blob_storage_service_name, blob_storage_service_api_key)  
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)  
    sas_urls = []  
    for file in files:  
        blob_name = f"{folder}/{file.filename}" if folder else file.filename  
        blob_client = blob_service_client.get_blob_client(container=blob_storage_container, blob=blob_name)  
        blob_client.upload_blob(file.file, overwrite=True)  
  
        sas_token = generate_blob_sas(  
            account_name=blob_service_client.account_name,  
            container_name=blob_storage_container,  
            blob_name=blob_name,  
            account_key=blob_service_client.credential.account_key,  
            permission=BlobSasPermissions(read=True),  
            expiry=datetime.utcnow() + timedelta(hours=1)  
        )  
        sas_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{blob_storage_container}/{blob_name}?{sas_token}"  
        sas_urls.append(sas_url)  

    return {"files": sas_urls} 
  

def generate_answer(question, content, openai_gpt_api_base, openai_gpt_api_key, openai_gpt_api_version, openai_gpt_model):
    max_attempts = 6
    max_backoff = 60
    system_prompt = """
    You are an intelligent assistant. 
    Use 'you' to refer to the individual asking the questions even if they ask with 'I'. 
    Sometimes the answer may be in a table.
    Focus the response on the intent of the users question. For example, if they ask "Who is", aim to respond with information about "Who" as opposed to "How to".
    Each source has a URL to an image followed by a colon and the actual information. 
    Use markdown format to display the image inline with the response using Mardown format.
    For every fact, always include a reference to the source of that fact, even if you used the source to infer the fact.
    Aim to be succint, but include any relevent information you find in the content such as special rules, legalities, restrictions or other relevent notes.
    Only answer the question using the source information below. 
    Do not make up an answer.
    """

    user_prompt = question + "\nSources:\n" + content

    gpt_client = AzureOpenAI(
        api_version=openai_gpt_api_version,
        azure_endpoint=openai_gpt_api_base,
        api_key=openai_gpt_api_key
    )
    
    counter = 0
    incremental_backoff = 1   # seconds to wait on throttline - this will be incremental backoff
    while True and counter < max_attempts:
        try:
            response = gpt_client.chat.completions.create(
                model=openai_gpt_model, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=openai_temperature,
                max_tokens=max_tokens,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )
            return response.choices[0].message.content

        except openai.APIError as ex:
            # Handlethrottling - code 429
            if str(ex.code) == "429":
                incremental_backoff = min(max_backoff, incremental_backoff * 1.5)
                print ('Waiting to retry after', incremental_backoff, 'seconds...')
                counter += 1
                time.sleep(incremental_backoff)
            elif str(ex.code) == "DeploymentNotFound":
                print ('Error: Deployment not found')
                return 'Error: Deployment not found'
            elif 'Error code: 40' in str(ex):
                print ('Error: ' + str(ex))
                return 'Error:' + str(ex)
            elif 'Connection error' in str(ex):
                print ('Error: Connection error')
                return 'Error: Connection error'                
            elif str(ex.code) == "content_filter":
                print ('Conten Filter Error', ex.code)
                return "Error: Content could not be extracted due to Azure OpenAI content filter." + ex.code
            else:
                print ('API Error:', ex)
                print ('API Error Code:', ex.code)
                incremental_backoff = min(max_backoff, incremental_backoff * 1.5)
                counter += 1
                time.sleep(incremental_backoff)
        except Exception as ex:
            counter += 1
            print ('Error - Retry count:', counter, ex)
        
        return ""
        

citation_pattern = r'\[([^\]]+)\]'  
def extract_citations(text):
    citations = re.findall(citation_pattern, answer)  
    return citations
    
@app.post("/chat", response_class=JSONResponse)  
async def chat(user_input: str = Form(...),  
                search_service_name: str = Form(...),  
                search_index_name: str = Form(...),  
                search_admin_key: str = Form(...),  
                search_api_version: str = Form("2024-05-01-preview"),
                openai_gpt_api_base: str = Form(...),
                openai_gpt_api_key: str = Form(...), 
                openai_gpt_api_version: str = Form(...),
                openai_gpt_model: str = Form(...),
                blob_storage_service_name: str = Form(...),
                blob_storage_service_api_key: str = Form(...),
                blob_storage_container: str = Form(...)
               ):  
    headers = {  
        "Content-Type": "application/json",  
        "api-key": search_admin_key  
    }  
    
    print ('User Input:', user_input)
    
    payload = {  
        "search": user_input,  
        "searchFields": "content",
        "select": "doc_id, pg_number, content",
        "top": 5
    }  
    url = f"https://{search_service_name}.search.windows.net/indexes/{search_index_name}/docs/search?api-version={search_api_version}"  
      
    async with httpx.AsyncClient() as client:  
        print('Searching...')
        response = await client.post(url, headers=headers, json=payload)  
        print('Checking response...')
        if response.status_code != 200:  
            print('Not 200:', response.status_code)
            raise HTTPException(status_code=response.status_code, detail="Failed to get response from Azure AI Search")  
  
        result = response.json() 
        print (result)
        if "value" in result and len(result["value"]) > 0:  
            # search_result = result["value"][0]["content"]  
            search_result = ''
            for result in result["value"]: 
                base_url, chunk_id, pg_number = parse_doc_id(result['doc_id'] + '-' + str(result['pg_number']))
                
                blob_name = f'processed/{base_url}/images/{pg_number}.png' 
                print ('Blob Name:', blob_name)

                sas_token = generate_blob_sas(  
                    account_name=blob_storage_service_name,  
                    container_name=blob_storage_container,  
                    blob_name=blob_name,  
                    account_key=blob_storage_service_api_key,  
                    permission=BlobSasPermissions(read=True),  
                    expiry=datetime.utcnow() + timedelta(hours=1)  
                )  
                sas_url = f"https://{blob_storage_service_name}.blob.core.windows.net/{blob_storage_container}/{blob_name}?{sas_token}"  

                print ('SAS URL:', sas_url)                
                
                search_result += sas_url + ': ' + result['content'] + '\n\n'
            print ('SEARCH RESULTS:', search_result)
            answer = generate_answer(user_input, search_result, openai_gpt_api_base, openai_gpt_api_key, openai_gpt_api_version, openai_gpt_model)   
            print ('ANSWER:', answer)
        else:  
            answer = "No results found."  
        
 
        return {"response": answer}  
    
  
@app.get("/", response_class=HTMLResponse)  
async def read_form(  
    request: Request, job_id: str = None, status: str = None,  
    blob_storage_service_name: str = "", blob_storage_service_api_key: str = "", blob_storage_container: str = "", folder: str = "",  
    openai_gpt_api_base: str = "", openai_gpt_api_key: str = "", openai_gpt_api_version: str = "", openai_gpt_model: str = "",  
    openai_embedding_api_base: str = "", openai_embedding_api_key: str = "", openai_embedding_api_version: str = "", openai_embedding_model: str = "",  
    search_service_name: str = "", search_index_name: str = "", search_admin_key: str = "", search_api_version: str = "2024-05-01-preview", 
    prompt: str = """Extract everything you see in this image to markdown. Convert all charts such as line, pie and bar charts to markdown tables and include a note that the numbers are approximate."""  
):  
    return templates.TemplateResponse("index.html", {  
        "request": request,  
        "job_id": job_id,  
        "status": status,  
        "blob_storage_service_name": blob_storage_service_name,  
        "blob_storage_service_api_key": blob_storage_service_api_key,  
        "blob_storage_container": blob_storage_container,  
        "folder": folder,  
        "openai_gpt_api_base": openai_gpt_api_base,  
        "openai_gpt_api_key": openai_gpt_api_key,  
        "openai_gpt_api_version": openai_gpt_api_version,  
        "openai_gpt_model": openai_gpt_model,  
        "openai_embedding_api_base": openai_embedding_api_base,  
        "openai_embedding_api_key": openai_embedding_api_key,  
        "openai_embedding_api_version": openai_embedding_api_version,  
        "openai_embedding_model": openai_embedding_model,  
        "search_service_name": search_service_name,  
        "search_index_name": search_index_name,  
        "search_admin_key": search_admin_key,  
        "search_api_version": search_api_version,  
        "prompt": prompt,  
        "sas_urls": None  
    })  

@app.post("/upload-settings-file", response_class=HTMLResponse)  
async def upload_settings_file(request: Request, file: UploadFile = File(...)):  
    content = await file.read()  
    settings = json.loads(content)  
  
    return templates.TemplateResponse("index.html", {  
        "request": request,  
        "job_id": None,  
        "status": None,  
        "blob_storage_service_name": settings.get("blob_storage_service_name", ""),  
        "blob_storage_service_api_key": settings.get("blob_storage_service_api_key", ""),  
        "blob_storage_container": settings.get("blob_storage_container", ""),  
        "folder": settings.get("folder", ""),  
        "openai_gpt_api_base": settings.get("openai_gpt_api_base", ""),  
        "openai_gpt_api_key": settings.get("openai_gpt_api_key", ""),  
        "openai_gpt_api_version": settings.get("openai_gpt_api_version", ""),  
        "openai_gpt_model": settings.get("openai_gpt_model", ""),  
        "openai_embedding_api_base": settings.get("openai_embedding_api_base", ""),  
        "openai_embedding_api_key": settings.get("openai_embedding_api_key", ""),  
        "openai_embedding_api_version": settings.get("openai_embedding_api_version", ""),  
        "openai_embedding_model": settings.get("openai_embedding_model", ""),  
        "search_service_name": settings.get("search_service_name", ""),  
        "search_index_name": settings.get("search_index_name", ""),  
        "search_admin_key": settings.get("search_admin_key", ""),  
        "search_api_version": settings.get("search_api_version", "2024-05-01-preview"),  
        "prompt": settings.get("prompt", """Extract everything you see in this image to markdown. Convert all charts such as line, pie and bar charts to markdown tables and include a note that the numbers are approximate."""),  
        "job_service_url": settings.get("job_service_url", ""),  
        "sas_urls": None  
    })  
    
@app.post("/create-index")  
async def create_index(request: Request):  
    data = await request.json()  
    search_service_name = data.get("search_service_name")  
    search_index_name = data.get("search_index_name")  
    search_admin_key = data.get("search_admin_key")  
    search_api_version = data.get("search_api_version")  
    index_schema = data.get("index_schema")  
    
    search_service_url = "https://{}.search.windows.net/".format(search_service_name)
    search_headers = {  
        'Content-Type': 'application/json',  
        'api-key': search_admin_key  
    }  
    
    # Making the POST requests to re-create the index  
    delete_url = f"{search_service_url}/indexes/{search_index_name}?api-version={search_api_version}"  
    response = requests.delete(delete_url, headers=search_headers)  
    if response.status_code == 204:  
        print(f"Index {search_index_name} deleted successfully.")  
    else:  
        print("Error deleting index, it may not exist.")  
    
    # The endpoint URL for creating the index  
    create_index_url = f"{search_service_url}/indexes?api-version={search_api_version}"  
    response = requests.post(create_index_url, headers=search_headers, json=index_schema)  
      
    # Check the response  
    if response.status_code == 201:  
        print(f"Index {search_index_name} created successfully.")  
    else:  
        print(f"Error creating index {search_index_name} :")  
        print(response.json())  
    
    return {"status": f"Index {search_index_name} created successfully."}   

    
@app.post("/create-embedding")  
async def create_embedding(request: Request):
    print ('Creating embedding...')
    data = await request.json()  
    openai_embedding_api_base = data.get("openai_embedding_api_base")  
    openai_embedding_api_key = data.get("openai_embedding_api_key")  
    openai_embedding_api_version = data.get("openai_embedding_api_version")  
    openai_embeddings_model = data.get("openai_embedding_model")  
    text = data.get("text")  
    
    if openai_embedding_api_base == "" or openai_embedding_api_key == "" or openai_embedding_api_version == "" or openai_embeddings_model == "":
        return {"embedding": None, "status": "fail", "message": "Missing OpenAI parameters"}   
    elif text == "":
        return {"embedding": None, "status": "fail", "message": "Missing text parameters"}   
    
    try: 
        emb = generate_embedding(text, openai_embedding_api_version, openai_embedding_api_base, openai_embedding_api_key, openai_embeddings_model)
        print ('Checking if embedding...')
        if isinstance(emb, list):
            return {"embedding": emb, "status": "success"}   
        else:
            return {"embedding": None, "status": "fail", "message": emb}   
        print ('Done checking if embedding...')
            
    except Exception as ex:
        print (ex)
        return {"embedding": None, "status": "fail", "message": ex}   
    

@app.post("/create-answer")  
async def create_answer(request: Request):
    print ('Creating answer...')
    data = await request.json()  
    openai_gpt_api_base = data.get("openai_gpt_api_base")  
    openai_gpt_api_key = data.get("openai_gpt_api_key")  
    openai_gpt_api_version = data.get("openai_gpt_api_version")  
    openai_gpt_model = data.get("openai_gpt_model")  
    question = data.get("question") 
    content = data.get("content") 
    
    if openai_gpt_api_base == "" or openai_gpt_api_key == "" or openai_gpt_api_version == "" or openai_gpt_model == "":
        return {"answer": None, "status": "fail", "message": "Missing OpenAI parameters"}   
    elif question == "" or content == "":
        return {"answer": None, "status": "fail", "message": "Missing question or content parameters"}   
    
    try: 
        answer = generate_answer(question, content, openai_gpt_api_base, openai_gpt_api_key, openai_gpt_api_version, openai_gpt_model)
        print ('Checking answer...')
        if 'Error:' in answer:
            return {"answer": None, "status": "fail", "message": answer}   
        else:
            return {"answer": answer, "status": "success"}   
            
    except Exception as ex:
        print (ex)
        return {"answer": None, "status": "fail", "message": ex}   


@app.post("/test-blob")  
async def test_blob(request: Request):
    print ('Checking blob service and container...')
    data = await request.json()  
    blob_storage_service_name = data.get("blob_storage_service_name")  
    blob_storage_service_api_key = data.get("blob_storage_service_api_key")  
    blob_storage_container = data.get("blob_storage_container")
    
    if blob_storage_service_name == "" or blob_storage_service_api_key == "" or blob_storage_container == "":
        return {"status": "fail", "message": "Missing Blob parameters"}   
    
    connection_string = (  
        f"DefaultEndpointsProtocol=https;"  
        f"AccountName={blob_storage_service_name};"  
        f"AccountKey={blob_storage_service_api_key};"  
        f"EndpointSuffix=core.windows.net"  
    )  
    
    try:  
        # Create the BlobServiceClient  
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)  

        # Check if the service exists by attempting to list containers  
        try:  
            blob_service_client.list_containers()  
            print("Blob service exists.")  

            # Check if the container exists  
            container_client = blob_service_client.get_container_client(blob_storage_container)  
            try:  
                container_client.get_container_properties()  
                print(f"Container '{blob_storage_container}' exists.")  
                return {"status": "success", "message": "Container and service were both found"} 
            except ResourceNotFoundError:  
                print(f"Container '{blob_storage_container}' does not exist. Creating container.")  
                container_client.create_container()  
                print(f"Container '{blob_storage_container}' created successfully.")  
                return {"status": "success", "message": "Service existed, but container needed to be created."} 
        except HttpResponseError as e:  
            print(f"Failed to list containers: {e.message}")  
            print("This may indicate the service does not exist or there is an authentication issue.")  
            return {"status": "fail", "message": f"Failed to list containers: {e.message}. This may indicate the service does not exist or there is an authentication issue."} 

    except Exception as e:  
        print(f"An error occurred: {e}")  
        return {"status": "fail", "message": f"An error occurred: {e}"} 


@app.post("/test-search")  
async def test_search(request: Request):
    print ('Checking search service and index...')
    data = await request.json()  
    search_service_name = data.get("search_service_name")  
    search_index_name = data.get("search_index_name")  
    search_admin_key = data.get("search_admin_key")
    search_api_version = data.get("search_api_version")
    
    if search_service_name == "" or search_index_name == "" or search_admin_key == "" or search_api_version == "":
        return {"status": "fail", "message": "Missing Search parameters"}   

    headers = {  
        "Content-Type": "application/json",  
        "api-key": search_admin_key  
    }  
      
    try:  
        # Check if the service exists 
        try:  
            async with httpx.AsyncClient() as client:  
                url = f"https://{search_service_name}.search.windows.net/indexes?api-version={search_api_version}"  
                response = await client.get(url, headers=headers)  
                print('Checking search service...')
                if response.status_code != 200:  
                    print(f"Failed to check for search service. Error code: ", response.status_code)  
                    return {"status": "fail", "message": f"Failed to check for search service. Error code: " +  str(response.status_code)} 

                # Check if the index exists  
                url = f"https://{search_service_name}.search.windows.net/indexes/{search_index_name}?api-version={search_api_version}"  
                response = await client.get(url, headers=headers)  
                print('Checking search index...')
                if response.status_code != 200:  
                    print(f"Cound not find search index: ", response.status_code)  
                    return {"status": "fail", "message": f"Could not find index {search_index_name}, please create it from the 'AI Search Setup tab'"} 
                else:
                    print ("Search service and index were both found")
                    return {"status": "success", "message": "Search service and index were both found"} 
        except HttpResponseError as e:  
            print(f"HttpResponseError: {e.message}")  
            return {"status": "fail", "message": f"HttpResponseError: {e.message}"} 

    except Exception as e:  
        print(f"An error occurred: {e}")  
        return {"status": "fail", "message": f"An error occurred: {e}"} 



#########################################
# Duplicated 
#########################################

# Function to generate vectors for text
def generate_embedding(text, openai_embedding_api_version, openai_embedding_api_base, openai_embedding_api_key, openai_embeddings_model):
    max_attempts = 6
    max_backoff = 60
    if text == None:
        return None
        
    client = AzureOpenAI(
        api_version=openai_embedding_api_version,
        azure_endpoint=openai_embedding_api_base,
        api_key=openai_embedding_api_key
    )    
    counter = 0
    incremental_backoff = 1   # seconds to wait on throttline - this will be incremental backoff
    while True and counter < max_attempts:
        try:
            response = client.embeddings.create(
                input=text,
                model=openai_embeddings_model
            )
            return json.loads(response.model_dump_json())["data"][0]['embedding']
            
            
        except OpenAIError as ex:
            # Handlethrottling - code 429
            if str(ex.code) == "429":
                print ('OpenAI Throttling Error =- Waiting to retry after', incremental_backoff, 'seconds...')
                incremental_backoff = min(max_backoff, incremental_backoff * 1.5)
                counter += 1
                time.sleep(incremental_backoff)
            elif str(ex.code) == "DeploymentNotFound":
                print ('Error: Deployment not found')
                return 'Error: Deployment not found'
            elif 'Error code: 40' in str(ex):
                print ('Error: ' + str(ex))
                return 'Error:' + str(ex)
            elif 'Connection error' in str(ex):
                print ('Error: Connection error')
                return 'Error: Connection error'                
            elif str(ex.code) == "content_filter":
                print ('Conten Filter Error', ex.code)
                return "Error: Content could not be extracted due to Azure OpenAI content filter." + ex.code
            else:
                print ('API Error:', ex)
                print ('API Error Code:', ex.code)
                incremental_backoff = min(max_backoff, incremental_backoff * 1.5)
                counter += 1
                time.sleep(incremental_backoff)
        except Exception as ex:
            counter += 1
            print ('Error - Retry count:', counter, ex)
            
    return None
    