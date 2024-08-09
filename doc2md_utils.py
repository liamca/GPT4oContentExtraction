#!/usr/bin/env python
# coding: utf-8

import os
from tenacity import retry, wait_random_exponential, stop_after_attempt 
import shutil  
import json

# Azure OpenAI
from openai import AzureOpenAI
import io
import base64
import requests

# Image extraction from PDF
import fitz  # PyMuPDF  
from pathlib import Path  
import uuid

# For LibreOffice Doc Conversion to PDF
import subprocess  
import pathlib


config = json.load(open("config.json"))

# Azure AI Search Config
search_service_name = config["search_service_name"]
search_service_url = "https://{}.search.windows.net/".format(search_service_name)

search_admin_key = config["search_admin_key"]

index_name = config["search_index_name"]
index_schema_file = config["search_index_schema_file"]
search_api_version = config["search_api_version"]
search_headers = {  
    'Content-Type': 'application/json',  
    'api-key': search_admin_key  
}  

#Azure OpenAI
openai_embedding_api_base = config["openai_embedding_api_base"]
openai_embedding_api_key = config["openai_embedding_api_key"]
openai_embedding_api_version = config["openai_embedding_api_version"]
openai_embeddings_model = config["openai_embedding_model"]

# gets the API Key from environment variable AZURE_OPENAI_API_KEY
embeddings_client = AzureOpenAI(
    api_version=openai_embedding_api_version,
    azure_endpoint=openai_embedding_api_base,
    api_key=openai_embedding_api_key
)

openai_gpt_api_base = config["openai_gpt_api_base"]
openai_gpt_api_key = config["openai_gpt_api_key"]
openai_gpt_api_version = config["openai_gpt_api_version"]
openai_gpt_model = config["openai_gpt_model"]

gpt_client = AzureOpenAI(
    api_key=openai_gpt_api_key,  
    api_version=openai_gpt_api_version,
    base_url=f"{openai_gpt_api_base}/openai/deployments/{openai_gpt_model}"
)

supported_conversion_types = ['.pptx', '.ppt', '.docx', '.doc', '.xlsx', '.xls', '.pdf']

print ('Search Service Name:', search_service_name)
print ('Index Name:', index_name)
print ('Azure OpenAI GPT Base URL:', openai_gpt_api_base)
print ('Azure OpenAI GPT Model:', openai_gpt_model)
print ('Azure OpenAI Embeddings Base URL:', openai_embedding_api_base)
print ('Azure OpenAI Embeddings Model:', openai_embeddings_model)


def reset_local_dirs():
    if os.path.exists('json'):
        remove_directory('json')
    if os.path.exists('images'):
        remove_directory('images')
    if os.path.exists('markdown'):
        remove_directory('markdown')
    if os.path.exists('pdf'):
        remove_directory('pdf')
    if os.path.exists('merged'):
        remove_directory('merged')
    if os.path.exists('tmp'):
        remove_directory('tmp')

# Create directory if it does not exist
def ensure_directory_exists(directory_path):  
    path = Path(directory_path)  
    if not path.exists():  
        path.mkdir(parents=True, exist_ok=True)  
        print(f"Directory created: {directory_path}")  
    else:  
        print(f"Directory already exists: {directory_path}")  
  
# Remove a dir and sub-dirs
def remove_directory(directory_path):  
    try:  
        if os.path.exists(directory_path):  
            shutil.rmtree(directory_path)  
            print(f"Directory '{directory_path}' has been removed successfully.")  
        else:  
            print(f"Directory '{directory_path}' does not exist.")  
    except Exception as e:  
        print(f"An error occurred while removing the directory: {e}")  
    
# Convert to PDF
def convert_to_pdf(input_path):  
    file_suffix = pathlib.Path(input_path).suffix.lower()
    
    if file_suffix in supported_conversion_types:
        ensure_directory_exists('pdf')  
        
        output_file = input_path.replace(pathlib.Path(input_path).suffix, '')
        output_file = os.path.join('pdf', output_file + '.pdf')
    
        print ('Converting', input_path, 'to', output_file)
        if os.path.exists(output_file):
            os.remove(output_file)
    
        if file_suffix == '.pdf':
            # No need to convert, just copy
            shutil.copy(input_path, output_file)  
        else:
            # Command to convert pptx to pdf using LibreOffice  
            command = [  
                'soffice',  # or 'libreoffice' depending on your installation  
                '--headless',  # Run LibreOffice in headless mode (no GUI)  
                '--convert-to', 'pdf',  # Specify conversion format  
                '--outdir', os.path.dirname(output_file),  # Output directory  
                input_path  # Input file  
            ]  
              
            # Run the command  
            subprocess.run(command, check=True)  
            print(f"Conversion complete: {output_file}")  
    else:
        print ('File type not supported.')  
        return ""
    
    return output_file

# Convert pages from PDF to images
def extract_pdf_pages_to_images(pdf_path, image_dir):
    # Validate image_out directory exists
    doc_id = str(uuid.uuid4())
    image_out_dir = os.path.join(image_dir, doc_id)
    ensure_directory_exists(image_out_dir)  

    # Open the PDF file and iterate pages
    print ('Extracting images from PDF...')
    pdf_document = fitz.open(pdf_path)  

    for page_number in range(len(pdf_document)):  
        page = pdf_document.load_page(page_number)  
        image = page.get_pixmap()  
        image_out_file = os.path.join(image_out_dir, f'{page_number + 1}.png')
        image.save(image_out_file)  
        if page_number % 100 == 0:
            print(f'Processed {page_number} images...')  

    return doc_id

# Base64 encode images
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
        
# Find all files in a dir
def get_all_files(directory_path):  
    files = []  
    for entry in os.listdir(directory_path):  
        entry_path = os.path.join(directory_path, entry)  
        if os.path.isfile(entry_path):  
            files.append(entry_path)  
    return files  
  
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def extract_markdown_from_image(image_path):
    try:
        base64_image = encode_image(image_path)
        response = gpt_client.chat.completions.create(
            model=openai_gpt_model,
            messages=[
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": [  
                    { 
                        "type": "text", 
                        "text": """Extract everything you see in this image to markdown. 
                            Convert all charts such as line, pie and bar charts to markdown tables and include a note that the numbers are approximate.
                        """ 
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ] } 
            ],
            max_tokens=2000 
        )
        return response.choices[0].message.content
    except Exception as ex:
        return ""

def process_image(file, markdown_out_dir):
    if '.png' in file:
        print ('Processing:', file)
        markdown_file_out = os.path.join(markdown_out_dir, os.path.basename(file).replace('.png', '.txt'))
        print(markdown_file_out)
        if os.path.exists(markdown_file_out) == False:
            markdown_text = extract_markdown_from_image(file)
            with open(markdown_file_out, 'w', encoding='utf-8') as md_out:
                md_out.write(markdown_text)
        else:
            print ('Skipping processed file.')
    else:
        print ('Skipping non PNG file:', file)

    return file
  
def extract_numeric_value(filename):  
    # Extract numeric value from filename using regular expression  
    match = re.search(r'(\d+)', filename)  
    return int(match.group(1)) if match else float('inf') 
    
#######################################
# Indexing Azure AI Search Utils
#######################################
def create_index():
    dims = len(generate_embedding('That quick brown fox.'))
    print ('Dimensions in Embedding Model:', dims)
    
    with open(index_schema_file, "r") as f_in:
        index_schema = json.loads(f_in.read())
        index_schema['name'] = index_name
        index_schema['vectorSearch']['vectorizers'][0]['azureOpenAIParameters']['resourceUri'] = openai_embedding_api_base
        index_schema['vectorSearch']['vectorizers'][0]['azureOpenAIParameters']['deploymentId'] = openai_embeddings_model
        index_schema['vectorSearch']['vectorizers'][0]['azureOpenAIParameters']['apiKey'] = openai_embedding_api_key
        index_schema['vectorSearch']['vectorizers'][0]['azureOpenAIParameters']['apiKey'] = openai_embedding_api_key
    
    # Making the POST requests to re-create the index  
    delete_url = f"{search_service_url}/indexes/{index_name}?api-version={search_api_version}"  
    response = requests.delete(delete_url, headers=search_headers)  
    if response.status_code == 204:  
        print(f"Index {index_name} deleted successfully.")  
        # print(json.dumps(response.json(), indent=2))  
    else:  
        print("Error deleting index, it may not exist.")  
    
    # The endpoint URL for creating the index  
    create_index_url = f"{search_service_url}/indexes?api-version={search_api_version}"  
    response = requests.post(create_index_url, headers=search_headers, json=index_schema)  
      
    # Check the response  
    if response.status_code == 201:  
        print(f"Index {index_name} created successfully.")  
        # print(json.dumps(response.json(), indent=2))  
    else:  
        print(f"Error creating index {index_name} :")  
        print(response.json())  


def extract_numeric_value(filename):  
    # Extract numeric value from filename using regular expression  
    match = re.search(r'(\d+)', filename)  
    return int(match.group(1)) if match else float('inf') 
        
# Function to generate vectors for title and content fields, also used for query vectors
max_attempts = 6
max_backoff = 60
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(max_attempts))
def generate_embedding(text):
    if text == None:
        return None
        
    if len(text) < 10:
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
            # text-embedding-3-small == 1536 dims
            response = client.embeddings.create(
                input=text,
                model=openai_embeddings_model
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


def process_json(file, doc_id, markdown_out_dir, json_out_dir):
    if '.txt' in file:
        with open(os.path.join(markdown_out_dir, file), 'r') as c_in:
            content = c_in.read()

        json_data = {
            'doc_id': doc_id, 
            'page_number': int(file.replace('page_', '').replace('.txt', '')),
            'content': content
            }

        json_data['vector'] = generate_embedding(json_data['content'])


        with open(os.path.join(json_out_dir, file.replace('.txt', '.json')), 'w') as c_out:
            c_out.write(json.dumps(json_data, indent=4))

    else:
        print ('Skipping non JSON file:', file)

    return file

def index_content(json_files):
    # Index the content
    batch_size = 50
    index_doc_url = f"{search_service_url}/indexes/{index_name}/docs/index?api-version={search_api_version}" 
    
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

def get_doc_id(dir_path):
    entries = os.listdir(dir_path)  
    directories = [entry for entry in entries if os.path.isdir(os.path.join(dir_path, entry))]  
    if len(directories) > 0:
        print ('doc_id:', directories[0])
        return directories[0]
    else:
        print ('Could not find most recent doc_id')
        return None
