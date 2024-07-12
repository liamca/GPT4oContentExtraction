import os
import glob  
import requests  
import time
import json
from azure.storage.blob import BlobServiceClient, ContainerClient  
from openai import AzureOpenAI
from pathlib import Path  
import concurrent.futures  
from functools import partial 

# Function to generate vectors for title and content fields, also used for query vectors
max_attempts = 6
max_backoff = 60
def generate_embedding(text, data):
    if text == None:
        return None

    client = AzureOpenAI(
        api_version=data['openai_embedding_api_version'],
        azure_endpoint=data['openai_embedding_api_base'],
        api_key=data['openai_embedding_api_key']
    )    
    counter = 0
    incremental_backoff = 1   # seconds to wait on throttline - this will be incremental backoff
    while True and counter < max_attempts:
        try:
            response = client.embeddings.create(
                input=text,
                model=data['openai_embedding_model']
            )
            return json.loads(response.model_dump_json())["data"][0]['embedding']
        except openai.APIError as ex:
            # Handlethrottling - code 429
            if str(ex.code) == "429":
                incremental_backoff = min(max_backoff, incremental_backoff * 1.5)
                print ('Waiting to retry after', incremental_backoff, 'seconds...')
                time.sleep(incremental_backoff)
            elif str(ex.code) == "content_filter":
                print ('API Error', ex.code)
                return None
        except Exception as ex:
            counter += 1
            print ('Error - Retry count:', counter, ex)
    return None
    
def create_index(data):
    search_headers = {  
        'Content-Type': 'application/json',  
        'api-key': data['search_admin_key']
    }  
    dims = len(generate_embedding('The quick brown fox.', data))
    print ('Dimensions in Embedding Model:', dims)
    
    with open("schema.json", "r") as f_in:
        index_schema = json.loads(f_in.read())
        index_schema['name'] = data['search_index_name']
        index_schema['vectorSearch']['vectorizers'][0]['azureOpenAIParameters']['resourceUri'] = data['openai_embedding_api_base']
        index_schema['vectorSearch']['vectorizers'][0]['azureOpenAIParameters']['deploymentId'] = data['openai_embedding_model']
        index_schema['vectorSearch']['vectorizers'][0]['azureOpenAIParameters']['apiKey'] = data['openai_embedding_api_key']

    # Making the POST requests to re-create the index  
    search_service_url = "https://{}.search.windows.net/".format(data['search_service_name'])
    delete_url = f"{search_service_url}/indexes/{data['search_index_name']}?api-version={data['search_api_version']}"  
    response = requests.delete(delete_url, headers=search_headers)  
    if response.status_code == 204:  
        print(f"Index {data['search_index_name']} deleted successfully.")  
        # print(json.dumps(response.json(), indent=2))  
    else:  
        print("Error deleting index, it may not exist.")  
    
    # The endpoint URL for creating the index  
    create_index_url = f"{search_service_url}/indexes?api-version={data['search_api_version']}"  
    response = requests.post(create_index_url, headers=search_headers, json=index_schema)  
      
    # Check the response  
    if response.status_code == 201:  
        print(f"Index {data['search_index_name']} created successfully.")  
        # print(json.dumps(response.json(), indent=2))  
    else:  
        print(f"Error creating index {data['search_index_name']} :")  
        print(response.json())

# Create directory if it does not exist
def ensure_directory_exists(directory_path):  
    path = Path(directory_path)  
    if not path.exists():  
        path.mkdir(parents=True, exist_ok=True)  
        print(f"Directory created: {directory_path}")  
    else:  
        print(f"Directory already exists: {directory_path}")  

def process_json(file, doc_id, json_out_dir, data):
    print ('file', file)
    if '.txt' in file:
        with open(file, 'r', encoding="utf8") as c_in:
            content = c_in.read()

        json_data = {
            'doc_id': doc_id, 
            'page_number': int(os.path.basename(file).replace('.txt', '')),
            'content': content
            }

        json_data['vector'] = generate_embedding(json_data['content'], data)


        with open(os.path.join(json_out_dir, os.path.basename(file).replace('.txt', '.json')), 'w') as c_out:
            c_out.write(json.dumps(json_data, indent=4))

    else:
        print ('Skipping non JSON file:', file)

    return file


def index_content(json_files, data):
    # Index the content
    batch_size = 50
    search_headers = {  
        'Content-Type': 'application/json',  
        'api-key': data['search_admin_key']
    }  
    search_service_url = "https://{}.search.windows.net/".format(data['search_service_name'])
    index_doc_url = f"{search_service_url}/indexes/{data['search_index_name']}/docs/index?api-version={data['search_api_version']}" 
    
    documents = {"value": []}
    for file in json_files:
        if '.json' in file:
            with open(file, 'r') as j_in:
                json_data = json.loads(j_in.read())
            json_data['doc_id'] = json_data['doc_id'] + '-' + str(json_data['page_number'])
            documents["value"].append(json_data)
            if len(documents["value"]) == batch_size:
                response = requests.post(index_doc_url, headers=search_headers, json=documents)  
                # Check the response  
                if response.status_code == 200:  
                    print(f"Document Indexed successfully.")  
                    # print(json.dumps(response.json(), indent=2))  
                else:  
                    print(f"Error indexing document {file} :")  
                    print(response.json())  
                documents = {"value": []}
                
    response = requests.post(index_doc_url, headers=search_headers, json=documents)  
    # Check the response  
    if response.status_code == 200:  
        print(f"Documents Indexed successfully.")  
        # print(json.dumps(response.json(), indent=2))  
    else:  
        print(f"Error indexing documents {file} :")  
        print(response.json())  
    documents = {"value": []}
