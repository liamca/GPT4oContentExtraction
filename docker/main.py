import os  
import re  
import json  
import time  
import shutil  
import base64  
import pathlib  
import subprocess  
import concurrent.futures  
from datetime import datetime, timedelta  
from pathlib import Path  
from typing import Optional, Dict, List  
import httpx  
import requests  
import pdfkit  
import fitz  # PyMuPDF  
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Form, File, UploadFile  
from fastapi.responses import HTMLResponse, JSONResponse  
from fastapi.templating import Jinja2Templates  
from pydantic import BaseModel  
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, generate_blob_sas, BlobSasPermissions  
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError  
from openai import AzureOpenAI, OpenAIError  
from langchain.text_splitter import TokenTextSplitter, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter  
import tiktoken  



# Initialize the tokenizer  
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  
  
# wkhtmltopdf settings  
path_to_wkhtmltopdf = '/usr/bin/wkhtmltopdf'  # Adjust this path accordingly  
wkhtmltopdf_options = {  
    'page-size': 'A4',  
    'margin-top': '0.75in',  
    'margin-right': '0.75in',  
    'margin-bottom': '0.75in',  
    'margin-left': '0.75in',  
    'encoding': "UTF-8",  
    'no-outline': None,  
    'print-media-type': None,  
    'javascript-delay': 1000,  # Add delay to ensure JS content is rendered  
    'no-stop-slow-scripts': True  # Allow long-running scripts  
}  
  
# Chunking Config 
text_chunk_size = 8192
text_chunk_overlap = 820

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

# MD splits
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=text_chunk_size, chunk_overlap=text_chunk_overlap
)

header_1_split = [("#", "Header 1")]  
header_2_split = [("#", "Header 1"), ("##", "Header 2")]  
header_3_split = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]  
markdown_splitter_header_1 = MarkdownHeaderTextSplitter(headers_to_split_on=header_1_split)  
markdown_splitter_header_2 = MarkdownHeaderTextSplitter(headers_to_split_on=header_2_split)  
markdown_splitter_header_3 = MarkdownHeaderTextSplitter(headers_to_split_on=header_3_split)  
supported_conversion_types = ['.pptx', '.ppt', '.docx', '.doc', '.xlsx', '.xls', '.png', 'jpg']  
# max_chunk_len = 2048  
# text_splitter = TokenTextSplitter(chunk_size=max_chunk_len, chunk_overlap=205)  
  
app = FastAPI()  
templates = Jinja2Templates(directory="templates")  
max_tokens = 2048
openai_temperature = 0.1  
  
# Define Pydantic models  
class JobRequest(BaseModel):  
    url_file_to_process: str  
    openai_gpt_api_base: str  
    openai_gpt_api_key: str  
    openai_gpt_api_version: str  
    openai_gpt_model: str  
    blob_storage_service_name: str  
    blob_storage_service_api_key: str  
    blob_storage_container: str  
    openai_embedding_api_base: Optional[str] = None  
    openai_embedding_api_key: Optional[str] = None  
    openai_embedding_api_version: Optional[str] = None  
    openai_embedding_model: Optional[str] = None  
    prompt: Optional[str] = None  
    search_service_name: Optional[str] = None  
    search_admin_key: Optional[str] = None  
    search_index_name: Optional[str] = None  
    search_api_version: Optional[str] = None  
    chunk_type: Optional[str] = None  
    is_html: Optional[bool] = False  # deprecated and no longer used  
  
class JobStatus(BaseModel):  
    job_id: str  
    blob_storage_service_name: str  
    blob_storage_service_api_key: str  
    blob_storage_container: str  
  
class UploadFiles(BaseModel):  
    blob_storage_service_name: str  
    blob_storage_service_api_key: str  
    blob_storage_container: str  
    folder: str  
  
# Utility functions  
def encode_base64(input_string):  
    byte_string = input_string.encode('utf-8')  
    encoded_bytes = base64.b64encode(byte_string)  
    encoded_string = encoded_bytes.decode('utf-8')  
    return encoded_string  
  
def decode_base64(encoded_string):  
    encoded_bytes = encoded_string.encode('utf-8')  
    decoded_bytes = base64.b64decode(encoded_bytes)  
    decoded_string = decoded_bytes.decode('utf-8')  
    return decoded_string  

def construct_connection_string(account_name: str, account_key: str) -> str:
    return f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"

def find_first_page_number(text):  
    pattern = r'\|\|(\d+)\|\|'  
    matches = re.findall(pattern, text)  
    if matches:  
        return int(matches[0])  
    else:  
        return None 

def find_last_heading_level_1(markdown_text):  
    lines = markdown_text.split('\n')  
    last_heading = None  
      
    for line in lines:  
        if line.startswith('# '):  
            last_heading = line  
      
    return last_heading  
  
def find_page_number(text):  
    pattern = r'\|\|(\d+)\|\|'  
    matches = re.findall(pattern, text)  
    if matches:  
        return int(matches[-1])  
    else:  
        return None  
      
def find_all_page_numbers(text):  
	pattern = r'\|\|(\d+)\|\|'  
	matches = re.findall(pattern, text)
	if matches:  
		return list(set(matches))  
	else:  
		return [0]  

      
def download_file(url, local_filename):  
    response = requests.get(url, stream=True)  
    with open(local_filename, 'wb') as file:  
        for chunk in response.iter_content(chunk_size=8192):  
            if chunk:  
                file.write(chunk)  
    return local_filename  
  
def ensure_directory_exists(directory_path):  
    path = Path(directory_path)  
    if not path.exists():  
        path.mkdir(parents=True, exist_ok=True)  
        print(f"Directory created: {directory_path}")  
    else:  
        print(f"Directory already exists: {directory_path}")  
  
def remove_directory(directory_path):  
    try:  
        if os.path.exists(directory_path):  
            shutil.rmtree(directory_path)  
            print(f"Directory '{directory_path}' has been removed successfully.")  
        else:  
            print(f"Directory '{directory_path}' does not exist.")  
    except Exception as e:  
        print(f"An error occurred while removing the directory: {e}")  
  
def convert_to_pdf(input_path, pdf_dir):  
    output_file = input_path.replace(pathlib.Path(input_path).suffix, '')  
    output_file = os.path.join(pdf_dir, os.path.basename(output_file + '.pdf'))  
    if os.path.exists(output_file):  
        os.remove(output_file)  
    command = [  
        'soffice',  # or 'libreoffice' depending on your installation  
        '--headless',  # Run LibreOffice in headless mode (no GUI)  
        '--convert-to', 'pdf',  # Specify conversion format  
        '--outdir', os.path.dirname(output_file),  # Output directory  
        input_path  # Input file  
    ]  
    subprocess.run(command, check=True)  
    print(f"Conversion complete: {output_file}")  
    return output_file  
  
def extract_pdf_pages_to_images(pdf_path, image_out_dir):  
    ensure_directory_exists(image_out_dir)  
    print('Extracting images from PDF...')  
    pdf_document = fitz.open(pdf_path)  
    pg_counter = 0  
    for page_number in range(len(pdf_document)):  
        page = pdf_document.load_page(page_number)  
        image = page.get_pixmap()  
        image_out_file = os.path.join(image_out_dir, f'{page_number + 1}.png')  
        image.save(image_out_file)  
        if page_number % 100 == 0:  
            print(f'Processed {page_number} images...')  
        pg_counter += 1  
    print(f'Processed {pg_counter}')  
    return image_out_dir  
  
def encode_image(image_path):  
    with open(image_path, "rb") as image_file:  
        return base64.b64encode(image_file.read()).decode("utf-8")  
  
def get_all_files(directory_path):  
    files = []  
    for entry in os.listdir(directory_path):  
        entry_path = os.path.join(directory_path, entry)  
        if os.path.isfile(entry_path):  
            files.append(entry_path)  
    return files  
  
def list_files_with_relative_paths(directory):  
    file_paths = []  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            relative_path = os.path.relpath(os.path.join(root, file), directory)  
            file_paths.append(relative_path)  
    return file_paths  
  
def extract_numeric_value(filename):  
    match = re.search(r'(\d+)', filename)  
    return int(match.group(1)) if match else float('inf')  
  
def upload_current_status(blob_container_client, job_dir, status, msg):  
    status['message'] = msg  
    blob_client = blob_container_client.get_blob_client(os.path.join(job_dir, 'status.json'))  
    blob_client.upload_blob(json.dumps(status, indent=4), overwrite=True)  
    return status  
  
def upload_files_to_blob_storage(blob_container_client, container_name, job_dir):  
    files = list_files_with_relative_paths(job_dir)  
    for file_path in files:  
        try:  
            blob_client = blob_container_client.get_blob_client(os.path.join(job_dir, file_path))  
            with open(os.path.join(job_dir, file_path), "rb") as data:  
                blob_client.upload_blob(data, overwrite=True)  
            print(f"Uploaded {file_path} to container {container_name}.")  
        except Exception as e:  
            print(f"Failed to upload {file_path}: {e}")  
  
def ensure_container(blob_container_client):  
    try:  
        blob_container_client.create_container()  
        print(f"Container created")  
    except Exception as e:  
        print(f"Container exists")  
  
def generate_embedding(text, openai_embedding_api_version, openai_embedding_api_base, openai_embedding_api_key, openai_embeddings_model):  
    max_attempts = 6  
    max_backoff = 60  
    if text is None:  
        return None  
    client = AzureOpenAI(  
        api_version=openai_embedding_api_version,  
        azure_endpoint=openai_embedding_api_base,  
        api_key=openai_embedding_api_key  
    )  
    counter = 0  
    incremental_backoff = 1  
    while True and counter < max_attempts:  
        try:  
            response = client.embeddings.create(  
                input=text,  
                model=openai_embeddings_model  
            )  
            return json.loads(response.model_dump_json())["data"][0]['embedding']  
        except OpenAIError as ex:  
            if str(ex.code) == "429":  
                print('OpenAI Throttling Error =- Waiting to retry after', incremental_backoff, 'seconds...')  
                incremental_backoff = min(max_backoff, incremental_backoff * 1.5)  
                counter += 1  
                time.sleep(incremental_backoff)  
            elif str(ex.code) == "DeploymentNotFound":  
                print('Error: Deployment not found')  
                return 'Error: Deployment not found'  
            elif 'Error code: 40' in str(ex):  
                print('Error: ' + str(ex))  
                return 'Error:' + str(ex)  
            elif 'Connection error' in str(ex):  
                print('Error: Connection error')  
                return 'Error: Connection error'  
            elif str(ex.code) == "content_filter":  
                print('Content Filter Error', ex.code)  
                return "Error: Content could not be extracted due to Azure OpenAI content filter." + ex.code  
            else:  
                print('API Error:', ex)  
                print('API Error Code:', ex.code)  
                incremental_backoff = min(max_backoff, incremental_backoff * 1.5)  
                counter += 1  
                time.sleep(incremental_backoff)  
        except Exception as ex:  
            counter += 1  
            print('Error - Retry count:', counter, ex)  
            return None  
  
# Background task functions  
def background_task(job_id: str, job_request: JobRequest):  
    try:  
        job_info = initialize_job(job_id, job_request)  
        blob_service_client, blob_container_client = setup_azure_connections(job_request)  
        job_dir, doc_dir, pdf_dir, image_dir = setup_directories(job_id)  
  
        job_info, download_file_name = download_and_convert_file(job_request, job_info, blob_container_client, job_dir, doc_dir, pdf_dir)  
        pdf_images_dir = extract_images_from_pdf(pdf_dir, image_dir, job_info, blob_container_client, job_dir)  
  
        markdown_dir = convert_images_to_markdown(pdf_images_dir, job_request, job_info, blob_container_client, job_dir)  
        job_info = upload_current_status(blob_container_client, job_dir, job_info, "Merging Markdown files.")  
        merged_markdown = merge_markdown_files(markdown_dir, job_dir, job_id)  
  
        if should_vectorize(job_request):  
            job_info = upload_current_status(blob_container_client, job_dir, job_info, "Vectorizing content to JSON.")  
            file_type = pathlib.Path(download_file_name).suffix.lower()  
            documents = vectorize_markdown(merged_markdown, markdown_dir, file_type, job_request, job_info, blob_container_client, job_dir, job_id)  
  
            if should_index(job_request):  
                index_documents(documents, job_request, job_info, blob_container_client, job_dir)  
  
        job_info = upload_current_status(blob_container_client, job_dir, job_info, "Uploading content to Blob Storage.")  
        upload_files_to_blob_storage(blob_container_client, job_request.blob_storage_container, job_dir)  
  
        clean_up(job_dir)  
  
        job_info['status'] = 'complete'  
        upload_current_status(blob_container_client, job_dir, job_info, "Processing complete.")  
    except Exception as ex:  
        handle_error(ex, job_info, blob_container_client, job_dir)  
  
def initialize_job(job_id: str, job_request: JobRequest) -> Dict:  
    job_info = {"job_id": job_id, "status": "in-progress", "message": "Initiating job and validating services..."}  
    ensure_directory_exists(job_id)  
    return job_info  
  
def setup_azure_connections(job_request: JobRequest):  
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={job_request.blob_storage_service_name};AccountKey={job_request.blob_storage_service_api_key};EndpointSuffix=core.windows.net"  
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)  
    blob_container_client = blob_service_client.get_container_client(job_request.blob_storage_container)  
    ensure_container(blob_container_client)  
    return blob_service_client, blob_container_client  
  
def setup_directories(job_id: str):  
    ensure_directory_exists('processed')  
    job_dir = os.path.join('processed', job_id)  
    ensure_directory_exists(job_dir)  
    doc_dir = os.path.join(job_dir, 'doc')  
    ensure_directory_exists(doc_dir)  
    pdf_dir = os.path.join(job_dir, 'pdf')  
    ensure_directory_exists(pdf_dir)  
    image_dir = os.path.join(job_dir, 'images')  
    ensure_directory_exists(image_dir)  
    return job_dir, doc_dir, pdf_dir, image_dir  
  
def download_and_convert_file(job_request: JobRequest, job_info: Dict, blob_container_client, job_dir: str, doc_dir: str, pdf_dir: str):  
    job_info = upload_current_status(blob_container_client, job_dir, job_info, "Downloading file for processing...")  
    download_file_name = job_request.url_file_to_process.split('?')[0]  
    download_file_name = os.path.basename(download_file_name)  
    download_path = os.path.join(doc_dir, download_file_name)  
    file_to_process = download_file(job_request.url_file_to_process, download_path)  
  
    if pathlib.Path(download_file_name).suffix.lower() in supported_conversion_types:  
        job_info = upload_current_status(blob_container_client, job_dir, job_info, "Converting file to PDF")  
        pdf_path = convert_to_pdf(file_to_process, pdf_dir)  
    elif download_file_name.endswith('.pdf'):  
        job_info = upload_current_status(blob_container_client, job_dir, job_info, "No conversion needed for PDF.")  
        shutil.copy(file_to_process, pdf_dir)  
        pdf_path = os.path.join(pdf_dir, os.path.basename(file_to_process))  
    else:  
        job_info = upload_current_status(blob_container_client, job_dir, job_info, "Converting HTML to PDF")  
        pdf_path = convert_html_to_pdf(job_request.url_file_to_process, pdf_dir, download_file_name)  
  
    return job_info, download_file_name  
  
def convert_html_to_pdf(url: str, pdf_dir: str, download_file_name: str) -> str:  
    pdf_path = os.path.join(pdf_dir, download_file_name + '.pdf')  
    config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)  
    pdfkit.from_url(url, pdf_path, options=wkhtmltopdf_options, configuration=config)  
    return pdf_path  
  
def extract_images_from_pdf(pdf_dir: str, image_dir: str, job_info: Dict, blob_container_client, job_dir: str) -> str:  
    pdf_path = os.path.join(pdf_dir, os.listdir(pdf_dir)[0])  
    job_info = upload_current_status(blob_container_client, job_dir, job_info, "Converting PDF pages to images...")  
    pdf_images_dir = extract_pdf_pages_to_images(pdf_path, image_dir)  
    return pdf_images_dir  
  
def convert_images_to_markdown(pdf_images_dir: str, job_request: JobRequest, job_info: Dict, blob_container_client, job_dir: str) -> str:  
    job_info = upload_current_status(blob_container_client, job_dir, job_info, "Converting images to Markdown")  
    markdown_dir = os.path.join(job_dir, 'markdown')  
    ensure_directory_exists(markdown_dir)  
    files = get_all_files(pdf_images_dir)  
    tasks = [(markdown_dir, job_request.openai_gpt_api_key, job_request.openai_gpt_api_version, job_request.openai_gpt_api_base, job_request.openai_gpt_model, file, job_request.prompt) for file in files]  
  
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  
        futures = [executor.submit(process_image, *task) for task in tasks]  
        for future in concurrent.futures.as_completed(futures):  
            result = future.result()  
            print(f'Result: {result}')  
  
    return markdown_dir  
  
def merge_markdown_files(markdown_dir: str, jobs_dir: str, job_id: str) -> str:  
    print('Merging markdown files to: ', markdown_dir)  
    files = os.listdir(markdown_dir)  
    txt_files = [f for f in files if f.endswith('.txt')]  
    print('Sorting markdown files...')  
    sorted_files = sorted(txt_files, key=extract_numeric_value)  
    print('Creating a merged markdown string...')  
    merged_markdown = ''  
    for f in sorted_files:  
        with open(os.path.join(markdown_dir, f), 'r') as f_in:  
            pg_number = extract_numeric_value(f)  
            merged_markdown += f'\n||{pg_number}||\n' + f_in.read() + '\n'  
    merged_markdown_dir = os.path.join(jobs_dir, 'merged_markdown')  
    ensure_directory_exists(merged_markdown_dir)  
    print('Writing merged markdown file...')  
    with open(os.path.join(merged_markdown_dir, job_id + '.txt'), 'w') as f_out:  
        f_out.write(merged_markdown)  
    return merged_markdown  
  
def should_vectorize(job_request: JobRequest) -> bool:  
    return all([job_request.openai_embedding_api_base, job_request.openai_embedding_api_key, job_request.openai_embedding_api_version, job_request.openai_embedding_model])  
  
def vectorize_markdown(merged_markdown: str, markdown_dir: str, file_type: str, job_request: JobRequest, job_info: Dict, blob_container_client, job_dir: str, job_id: str) -> list:  
    job_info = upload_current_status(blob_container_client, job_dir, job_info, "Vectorizing markdown to JSON")  
    print("Vectorizing markdown to JSON...")  
    json_dir = os.path.join(job_dir, 'json')  
    ensure_directory_exists(json_dir)  
  
    documents = []  
    chunk_strategy = "markdown"  
    if job_request.chunk_type == "page":  
        chunk_strategy = "page"  
    elif job_request.chunk_type == "markdown":  
        chunk_strategy = "markdown"  
    elif file_type in ['.pptx', '.ppt', '.xlsx', '.xls', '.png', 'jpg']:  
        chunk_strategy = "page"  
  
    if chunk_strategy == "page":  
        job_info = upload_current_status(blob_container_client, job_dir, job_info, "Vectorizing by Page chunks.")  
        documents = vectorize_by_page(merged_markdown, markdown_dir, job_request, job_id)  
    else:  
        job_info = upload_current_status(blob_container_client, job_dir, job_info, "Vectorizing by Markdown heading chunks.")  
        documents = vectorize_by_markdown(merged_markdown, job_request, job_id)  
  
    job_info = upload_current_status(blob_container_client, job_dir, job_info, "Completed vectorizing chunks.")  
    print("Saving JSON locally...")  
    with open(os.path.join(json_dir, job_id + '.json'), 'w') as j_out:  
        j_out.write(json.dumps(documents))  
  
    return documents  
  
def vectorize_by_page(merged_markdown: str, markdown_dir: str, job_request: JobRequest, job_id: str) -> list:  
    documents = []  
    print("Vectorizing by Page chunks...")  
    sorted_files = sorted(os.listdir(markdown_dir), key=extract_numeric_value)  
  
    for f in sorted_files:  
        print('Creating JSON document for: ', os.path.join(markdown_dir, f))  
        with open(os.path.join(markdown_dir, f), 'r') as f_in:  
            content = f_in.read()  
            pg_number = extract_numeric_value(f)  
            content = '||' + str(pg_number) + '||\n' + content

            json_data = {  
                "doc_id": f"{job_id}-{pg_number}",  
                "chunk_id": int(pg_number),  
                "file_name": os.path.basename(job_request.url_file_to_process),  
                "content": content,  
                "vector": generate_embedding(content, job_request.openai_embedding_api_version, job_request.openai_embedding_api_base, job_request.openai_embedding_api_key, job_request.openai_embedding_model)  
            }  
            documents.append(json_data)  
  
    return documents  

def vectorize_by_markdown(merged_markdown: str, job_request: JobRequest, job_id: str) -> list:  
    print("Vectorizing by Markdown heading chunks...")  
    documents = []  
    md_header_splits = markdown_splitter.split_text(merged_markdown)
    # Char-level splits
    splits = text_splitter.split_documents(md_header_splits)
    chunk_id = 0  
    pg_number = 1  
    
    last_heading = ''
    
    for s in splits:
        # Add the page number to start of chunk 
        text = s.page_content
        
        get_pg_number  = find_first_page_number(text) 
        if text.find('||') != 0:
            if get_pg_number != None:
                pg_number = get_pg_number
        text = '||' + str(pg_number-1) + '||\n' + last_heading + '\n' + text
        
        json_data = {  
            "doc_id": f"{job_id}-{chunk_id}",  
            "chunk_id": chunk_id,  
            "file_name": os.path.basename(job_request.url_file_to_process),  
            "content": text,  
            "title": last_heading,  
            "vector": generate_embedding(str(text), job_request.openai_embedding_api_version, job_request.openai_embedding_api_base, job_request.openai_embedding_api_key, job_request.openai_embedding_model)  
        }  
        chunk_id += 1  
        documents.append(json_data) 

        
        find_heading = find_last_heading_level_1(text)  
        if find_heading != None:
            last_heading = find_heading

    return documents  
  
def should_index(job_request: JobRequest) -> bool:  
    return all([job_request.search_service_name, job_request.search_admin_key, job_request.search_index_name, job_request.search_api_version])  
  
def index_documents(documents: list, job_request: JobRequest, job_info: Dict, blob_container_client, job_dir: str):  
    job_info = upload_current_status(blob_container_client, job_dir, job_info, "Indexing document to Azure AI Search")  
    search_headers = {'Content-Type': 'application/json', 'api-key': job_request.search_admin_key}  
    exists_url = f'https://{job_request.search_service_name}.search.windows.net/indexes/{job_request.search_index_name}?api-version={job_request.search_api_version}'  
  
    response = requests.get(exists_url, headers=search_headers)  
    if response.status_code != 200:  
        raise Exception("Azure AI Search Index does not exist, please create it.")  
  
    index_doc_url = f"https://{job_request.search_service_name}.search.windows.net/indexes/{job_request.search_index_name}/docs/index?api-version={job_request.search_api_version}"  
    response = requests.post(index_doc_url, headers=search_headers, json={"value": documents})  
    if response.status_code != 200:  
        raise Exception(f"Error indexing documents: {response.json()}")  
  
def clean_up(job_dir: str):  
    remove_directory(job_dir)  
  
def handle_error(ex: Exception, job_info: Dict, blob_container_client, job_dir: str):  
    print(ex)  
    job_info['status'] = 'error'  
    upload_current_status(blob_container_client, job_dir, job_info, f"Error: {str(ex)}")  
  
def extract_markdown_from_image(gpt_client, openai_gpt_model, image_path, prompt):  
    max_attempts = 6  
    max_backoff = 60  
    counter = 0  
    incremental_backoff = 1  
    try:  
        base64_image = encode_image(image_path)  
        print('Length of Base64 Image:', len(base64_image))  
        user_prompt = """Extract everything you see in this image to markdown.  
                         Convert all charts such as line, pie and bar charts to markdown tables and include a note that the numbers are approximate.  
                      """  
        if prompt is not None:  
            user_prompt = prompt  
  
        while True and counter < max_attempts:  
            response = gpt_client.chat.completions.create(  
                model=openai_gpt_model,  
                messages=[  
                    {"role": "system", "content": "You are a helpful assistant."},  
                    {"role": "user", "content": [  
                        {"type": "text", "text": user_prompt},  
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}  
                    ]}  
                ],  
                max_tokens=4096  
            )  
            print('Received Response')  
            return response.choices[0].message.content  
    except OpenAIError as ex:  
        if str(ex.code) == "429":  
            print('OpenAI Throttling Error =- Waiting to retry after', incremental_backoff, 'seconds...')  
            incremental_backoff = min(max_backoff, incremental_backoff * 1.5)  
            counter += 1  
            time.sleep(incremental_backoff)  
        elif str(ex.code) == "content_filter":  
            print('Content Filter Error', ex.code)  
            return "Content could not be extracted due to Azure OpenAI content filter." + ex.code  
        else:  
            print('API Error:', ex, ex.code)  
            incremental_backoff = min(max_backoff, incremental_backoff * 1.5)  
            counter += 1  
            time.sleep(incremental_backoff)  
    except Exception as ex:  
        counter += 1  
        print('Error - Retry count:', counter, ex)  
        return ""  
  
def process_image(markdown_out_dir, openai_gpt_api_key, openai_gpt_api_version, openai_gpt_api_base, openai_gpt_model, file, prompt):  
    if '.png' in file:  
        print('Processing:', file)  
        gpt_client = AzureOpenAI(  
            api_key=openai_gpt_api_key,  
            api_version=openai_gpt_api_version,  
            base_url=f"{openai_gpt_api_base}/openai/deployments/{openai_gpt_model}"  
        )  
        markdown_text = extract_markdown_from_image(gpt_client, openai_gpt_model, file, prompt)  
        markdown_file_out = os.path.join(markdown_out_dir, os.path.basename(file).replace('.png', '.txt'))  
        print(markdown_file_out), prompt  
        with open(markdown_file_out, 'w') as md_out:  
            md_out.write(markdown_text)  
    else:  
        print('Skipping non PNG file:', file)  
    return file  
  
# API Endpoints  
@app.post("/start-job")  
async def start_job(job_request: JobRequest, background_tasks: BackgroundTasks):  
    try:  
        download_url = job_request.url_file_to_process  
        if '?' in download_url:  
            download_url = download_url[:download_url.find('?')]  
        job_id = encode_base64(download_url)  
        job_info = {"job_id": job_id}  
        background_tasks.add_task(background_task, job_id, job_request)  
        return job_info  
    except Exception as ex:  
        job_info['status'] = "error"  
        print('Error:', ex)  
        job_info['status'] = "error"  
        job_info = upload_current_status(blob_container_client, job_dir, job_info, ex)  
  
@app.post("/job-status")  
async def job_status(job_status: JobStatus):  
    job_id = job_status.job_id  
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={job_status.blob_storage_service_name};AccountKey={job_status.blob_storage_service_api_key};EndpointSuffix=core.windows.net"  
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)  
    blob_name = os.path.join('processed', job_id, 'status.json')  
    print(blob_name)  
    blob_client = blob_service_client.get_blob_client(container=job_status.blob_storage_container, blob=blob_name)  
    job_info = json.loads(blob_client.download_blob().readall())  
    return job_info  
  
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

def remove_between_pipes(s):  
    # Split the string into lines  
    lines = s.split('\n')  
    # Filter out lines that start with ||  
    filtered_lines = [line for line in lines if not line.strip().startswith('||')]  
    # Join the remaining lines back into a single string  
    result = '\n'.join(filtered_lines)  
    return result  
	
def generate_answer(question, content, openai_gpt_api_base, openai_gpt_api_key, openai_gpt_api_version, openai_gpt_model):  
    max_attempts = 6  
    max_backoff = 60  
    system_prompt = """  
    You are an intelligent assistant.  
    Use 'you' to refer to the individual asking the questions even if they ask with 'I'.  
    Sometimes the answer may be in a table.  
    Only answer the question using the source information provided.  
    Do not make up an answer.  
    Your response should be in Markdown format.
    Before each section of text there is a URL to an image separated by || that is the source for that information. 
    Whenever you use information from the source, always reference the source image immedicately above the text.
    You should reference the image in the format [image_name.png](url).
    For example, if the source content was:

     	||https://xyx.com/foo/img1.png?sv=123||
        The captial of Canada is Ottawa
	||https://xyx.com/foo/img2.png?sv=321||
 	The captital of USA is Washington, DC
	||https://xyx.com/foo/img3.png?sv=987||

    And you used the source text "The captial of Canada is Ottawa", you would add as a source URL the URL which was located above this text as follows:

    [img1.png](https://xyx.com/foo/img1.png?sv=123)
        
    It is really important that you reference the image URL ABOVE the text you used and NOT the image URL below the text. 
    """  
    user_prompt = question + "\nSources:\n" + content  
    gpt_client = AzureOpenAI(  
        api_version=openai_gpt_api_version,  
        azure_endpoint=openai_gpt_api_base,  
        api_key=openai_gpt_api_key  
    )  
    counter = 0  
    incremental_backoff = 1  
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
            if str(ex.code) == "429":  
                incremental_backoff = min(max_backoff, incremental_backoff * 1.5)  
                print('Waiting to retry after', incremental_backoff, 'seconds...')  
                counter += 1  
                time.sleep(incremental_backoff)  
            elif str(ex.code) == "DeploymentNotFound":  
                print('Error: Deployment not found')  
                return 'Error: Deployment not found'  
            elif 'Error code: 40' in str(ex):  
                print('Error: ' + str(ex))  
                return 'Error:' + str(ex)  
            elif 'Connection error' in str(ex):  
                print('Error: Connection error')  
                return 'Error: Connection error'  
            elif str(ex.code) == "content_filter":  
                print('Content Filter Error', ex.code)  
                return "Error: Content could not be extracted due to Azure OpenAI content filter." + ex.code  
            else:  
                print('API Error:', ex)  
                print('API Error Code:', ex.code)  
                incremental_backoff = min(max_backoff, incremental_backoff * 1.5)  
                counter += 1  
                time.sleep(incremental_backoff)  
        except Exception as ex:  
            counter += 1  
            print('Error - Retry count:', counter, ex)  
            return ""  
  
citation_pattern = r'\[([^\]]+)\]'  
  
def extract_citations(text):  
    citations = re.findall(citation_pattern, text)  
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
  
    print('User Input:', user_input)  
  
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
        print(result)  
        if "value" in result and len(result["value"]) > 0:  
            search_result = ''  
            for result in result["value"]:  
                base_url, chunk_id, pg_number = parse_doc_id(result['doc_id'] + '-0')  
                result_content = result['content']
                pg_numbers = find_all_page_numbers(result_content)
                # sas_urls = []
                for pg in pg_numbers:
                    base_url, chunk_id, pg_number = parse_doc_id(result['doc_id'] + '-' + str(pg))  
                    blob_name = f'processed/{base_url}/images/{pg}.png'  
                    print('Blob Name:', blob_name)  
                    sas_token = generate_blob_sas(  
                        account_name=blob_storage_service_name,  
                        container_name=blob_storage_container,  
                        blob_name=blob_name,  
                        account_key=blob_storage_service_api_key,  
                        permission=BlobSasPermissions(read=True),  
                        expiry=datetime.utcnow() + timedelta(hours=1)  
                    )  
                    sas_url = f"https://{blob_storage_service_name}.blob.core.windows.net/{blob_storage_container}/{blob_name}?{sas_token}"  
                    print('SAS URL:', sas_url)  
                    result_content = result_content.replace('||' + str(pg) + '||', '||' + sas_url + '||') 
                    # sas_urls.append(sas_url)
  
                # search_result += str(sas_urls) + ': ' + result['content'] + '\n\n'  
                search_result += result_content + '\n\n'  
		    
            print('SEARCH RESULTS:', search_result)  
            answer = generate_answer(user_input, search_result, openai_gpt_api_base, openai_gpt_api_key, openai_gpt_api_version, openai_gpt_model)  
            # Sometimes GPT sends back the ||img|| - remove them manually
            answer = remove_between_pipes(answer)	
		
            print('ANSWER:', answer)  
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

    if index_schema == "template":
        index_schema = index_schema_template
        index_schema["name"] = search_index_name
  
    search_service_url = "https://{}.search.windows.net/".format(search_service_name)  
    search_headers = {  
        'Content-Type': 'application/json',  
        'api-key': search_admin_key  
    }  
  
    delete_url = f"{search_service_url}/indexes/{search_index_name}?api-version={search_api_version}"  
    response = requests.delete(delete_url, headers=search_headers)  
    if response.status_code == 204:  
        print(f"Index {search_index_name} deleted successfully.")  
    else:  
        print("Error deleting index, it may not exist.")  
  
    create_index_url = f"{search_service_url}/indexes?api-version={search_api_version}"  
    response = requests.post(create_index_url, headers=search_headers, json=index_schema)  
  
    if response.status_code == 201:
        print(f"Index {search_index_name} created successfully.")
        return {"status": f"Index {search_index_name} created successfully."}
    else:
        print(f"Error creating index {search_index_name} :")
        print(response.json())
        return {"status": f"Error Creating Index {search_index_name}."}

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


@app.post("/create-embedding")  
async def create_embedding(request: Request):  
    print('Creating embedding...')  
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
        print('Checking if embedding...')  
        if isinstance(emb, list):  
            return {"embedding": emb, "status": "success"}  
        else:  
            return {"embedding": None, "status": "fail", "message": emb}  
        print('Done checking if embedding...')  
    except Exception as ex:  
        print(ex)  
        return {"embedding": None, "status": "fail", "message": ex}  
  
@app.post("/create-answer")  
async def create_answer(request: Request):  
    print('Creating answer...')  
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
        print('Checking answer...')  
        if 'Error:' in answer:  
            return {"answer": None, "status": "fail", "message": answer}  
        else:  
            return {"answer": answer, "status": "success"}  
    except Exception as ex:  
        print(ex)  
        return {"answer": None, "status": "fail", "message": ex}  
  
@app.post("/test-blob")  
async def test_blob(request: Request):  
    print('Checking blob service and container...')  
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
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)  
        try:  
            blob_service_client.list_containers()  
            print("Blob service exists.")  
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
    print('Checking search service and index...')  
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
        try:  
            async with httpx.AsyncClient() as client:  
                url = f"https://{search_service_name}.search.windows.net/indexes?api-version={search_api_version}"  
                response = await client.get(url, headers=headers)  
                print('Checking search service...')  
                if response.status_code != 200:  
                    print(f"Failed to check for search service. Error code: ", response.status_code)  
                    return {"status": "fail", "message": f"Failed to check for search service. Error code: " + str(response.status_code)}  
  
                url = f"https://{search_service_name}.search.windows.net/indexes/{search_index_name}?api-version={search_api_version}"  
                response = await client.get(url, headers=headers)  
                print('Checking search index...')  
                if response.status_code != 200:  
                    print(f"Could not find search index: ", response.status_code)  
                    return {"status": "fail", "message": f"Could not find index {search_index_name}, please create it from the 'AI Search Setup tab'"}  
                else:  
                    print("Search service and index were both found")  
                    return {"status": "success", "message": "Search service and index were both found"}  
        except HttpResponseError as e:  
            print(f"HttpResponseError: {e.message}")  
            return {"status": "fail", "message": f"HttpResponseError: {e.message}"}  
    except Exception as e:  
        print(f"An error occurred: {e}")  
        return {"status": "fail", "message": f"An error occurred: {e}"}  

index_schema_template = {
  "name": "<redacted>",
  "defaultScoringProfile": None,
  "fields": [
    {
      "name": "doc_id",
      "type": "Edm.String",
      "searchable": False,
      "filterable": True,
      "retrievable": True,
      "stored": True,
      "sortable": True,
      "facetable": False,
      "key": True,
      "indexAnalyzer": None,
      "searchAnalyzer": None,
      "analyzer": None,
      "normalizer": None,
      "dimensions": None,
      "vectorSearchProfile": None,
      "vectorEncoding": None,
      "synonymMaps": []
    },
    {
      "name": "chunk_id",
      "type": "Edm.Int32",
      "searchable": False,
      "filterable": True,
      "retrievable": True,
      "stored": True,
      "sortable": True,
      "facetable": False,
      "key": False,
      "indexAnalyzer": None,
      "searchAnalyzer": None,
      "analyzer": None,
      "normalizer": None,
      "dimensions": None,
      "vectorSearchProfile": None,
      "vectorEncoding": None,
      "synonymMaps": []
    },{
      "name": "pg_number",
      "type": "Edm.Int32",
      "searchable": False,
      "filterable": True,
      "retrievable": True,
      "stored": True,
      "sortable": True,
      "facetable": False,
      "key": False,
      "indexAnalyzer": None,
      "searchAnalyzer": None,
      "analyzer": None,
      "normalizer": None,
      "dimensions": None,
      "vectorSearchProfile": None,
      "vectorEncoding": None,
      "synonymMaps": []
    },
    {
      "name": "file_name",
      "type": "Edm.String",
      "searchable": True,
      "filterable": True,
      "retrievable": True,
      "stored": True,
      "sortable": False,
      "facetable": False,
      "key": False,
      "indexAnalyzer": None,
      "searchAnalyzer": None,
      "analyzer": None,
      "normalizer": None,
      "dimensions": None,
      "vectorSearchProfile": None,
      "vectorEncoding": None,
      "synonymMaps": []
    },
    {
      "name": "title",
      "type": "Edm.String",
      "searchable": True,
      "filterable": False,
      "retrievable": True,
      "stored": True,
      "sortable": False,
      "facetable": False,
      "key": False,
      "indexAnalyzer": None,
      "searchAnalyzer": None,
      "analyzer": "en.microsoft",
      "normalizer": None,
      "dimensions": None,
      "vectorSearchProfile": None,
      "vectorEncoding": None,
      "synonymMaps": []
    },
    {
      "name": "content",
      "type": "Edm.String",
      "searchable": True,
      "filterable": False,
      "retrievable": True,
      "stored": True,
      "sortable": False,
      "facetable": False,
      "key": False,
      "indexAnalyzer": None,
      "searchAnalyzer": None,
      "analyzer": "en.microsoft",
      "normalizer": None,
      "dimensions": None,
      "vectorSearchProfile": None,
      "vectorEncoding": None,
      "synonymMaps": []
    },
    {
      "name": "vector",
      "type": "Collection(Edm.Single)",
      "searchable": True,
      "filterable": False,
      "retrievable": True,
      "stored": True,
      "sortable": False,
      "facetable": False,
      "key": False,
      "indexAnalyzer": None,
      "searchAnalyzer": None,
      "analyzer": None,
      "normalizer": None,
      "dimensions": 1536,
      "vectorSearchProfile": "vector-profile",
      "vectorEncoding": None,
      "synonymMaps": []
    }
  ],
  "scoringProfiles": [],
  "corsOptions": None,
  "suggesters": [],
  "analyzers": [],
  "normalizers": [],
  "tokenizers": [],
  "tokenFilters": [],
  "charFilters": [],
  "encryptionKey": None,
  "similarity": {
    "@odata.type": "#Microsoft.Azure.Search.BM25Similarity",
    "k1": None,
    "b": None
  },
  "semantic": {
    "defaultConfiguration": "vector-semantic-configuration",
    "configurations": [
      {
        "name": "vector-semantic-configuration",
        "prioritizedFields": {
          "titleField": {
            "fieldName": "title"
          },
          "prioritizedContentFields": [
            {
              "fieldName": "content"
            }
          ],
          "prioritizedKeywordsFields": []
        }
      }
    ]
  },
  "vectorSearch": {
    "algorithms": [
      {
        "name": "vector-algorithm",
        "kind": "hnsw",
        "hnswParameters": {
          "metric": "cosine",
          "m": 4,
          "efConstruction": 400,
          "efSearch": 500
        },
        "exhaustiveKnnParameters": None
      }
    ],
    "profiles": [
      {
        "name": "vector-profile",
        "algorithm": "vector-algorithm",
        "vectorizer": "vector-vectorizer",
        "compression": None
      }
    ],
    "vectorizers": [
      {
        "name": "vector-vectorizer",
        "kind": "azureOpenAI",
        "azureOpenAIParameters": {
          "resourceUri": "https://redacted.openai.azure.com/",
          "deploymentId": "redacted",
          "apiKey": "redacted",
          "modelName": "experimental",
          "authIdentity": None
        },
        "customWebApiParameters": None,
        "aiServicesVisionParameters": None,
        "amlParameters": None
      }
    ],
    "compressions": []
  }
}
