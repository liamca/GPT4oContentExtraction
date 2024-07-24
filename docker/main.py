from fastapi import FastAPI, BackgroundTasks, HTTPException  
from fastapi.responses import JSONResponse  
from pydantic import BaseModel  
from typing import Optional, Dict  
import os  
import shutil  
import re  
import json  
import time  
import requests  
import concurrent.futures  
import base64  
import subprocess  
import pathlib  
import pdfkit  
import fitz  # PyMuPDF  
from pathlib import Path  
from openai import AzureOpenAI, OpenAIError  
from tenacity import retry, wait_random_exponential, stop_after_attempt  
import io  
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient  
from langchain.text_splitter import TokenTextSplitter, MarkdownHeaderTextSplitter  
import copy  
import tiktoken  

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

#wkhtmltopdf settings
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
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=52)  
header_1_split = [
    ("#", "Header 1"),
]
header_2_split = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]
header_3_split = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter_header_1 = MarkdownHeaderTextSplitter(headers_to_split_on=header_1_split)
markdown_splitter_header_2 = MarkdownHeaderTextSplitter(headers_to_split_on=header_2_split)
markdown_splitter_header_3 = MarkdownHeaderTextSplitter(headers_to_split_on=header_3_split)

supported_conversion_types = ['.pptx', '.ppt', '.docx', '.doc', '.xlsx', '.xls', '.png', 'jpg']
max_chunk_len = 1024

app = FastAPI()  

# Define a Pydantic model for the request body with optional fields  
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
    is_html: Optional[bool] = False # depracated and no longer userd


class JobStatus(BaseModel):  
    job_id: str
    blob_storage_service_name: str
    blob_storage_service_api_key: str
    blob_storage_container: str

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
    print ('Merging markdown files to: ', markdown_dir)
    files = os.listdir(markdown_dir)  
    txt_files = [f for f in files if f.endswith('.txt')]  
    print ('Sorting markdown files...')
    sorted_files = sorted(txt_files, key=extract_numeric_value)  

    print ('Creating a merged markdown string...')
    merged_markdown = ''  
    for f in sorted_files:  
        with open(os.path.join(markdown_dir, f), 'r') as f_in:  
            pg_number = extract_numeric_value(f)  
            merged_markdown += f'\n||{pg_number}||\n' + f_in.read() + '\n'  
      
    merged_markdown_dir = os.path.join(jobs_dir, 'merged_markdown')  
    ensure_directory_exists(merged_markdown_dir)  
    print ('Writing merged markdown file...')
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
    if file_type in ['.pptx', '.ppt', '.xlsx', '.xls', '.png', 'jpg'] or job_request.chunk_type == "page":  
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
        print ('Creating JSON document for: ', os.path.join(markdown_dir, f))
        with open(os.path.join(markdown_dir, f), 'r') as f_in:  
            content = f_in.read()  
            pg_number = find_page_number(content) or extract_numeric_value(f)  
            json_data = {  
                "doc_id": f"{job_id}-{pg_number}",  
                "chunk_id": int(pg_number),  
                "pg_number": int(pg_number),  
                "file_name": os.path.basename(job_request.url_file_to_process),  
                "content": content,  
                "vector": generate_embedding(content, job_request.openai_embedding_api_version, job_request.openai_embedding_api_base, job_request.openai_embedding_api_key, job_request.openai_embedding_model)  
            }  
            documents.append(json_data)  
      
    return documents  
  
def vectorize_by_markdown(merged_markdown: str, job_request: JobRequest, job_id: str) -> list:  
    print("Vectorizing by Markdown heading chunks...")
    documents = []  
    header_1_splits = markdown_splitter_header_1.split_text(merged_markdown)  
    chunk_id = 0  
    pg_number = 1  
      
    for s1 in header_1_splits:  
        section_content = s1.page_content  
        token_count = len(encoding.encode(section_content))  
          
        if token_count > max_chunk_len:  
            header_2_splits = markdown_splitter_header_2.split_text(section_content)  
            for s2 in header_2_splits:  
                sub_section_content = ''  
                if 'Header 1' in s1.metadata:  
                    sub_section_content += '# ' + s1.metadata['Header 1'] + '\n'  
                if 'Header 2' in s2.metadata:  
                    sub_section_content += '## ' + s2.metadata['Header 2'] + '\n'  
                title = sub_section_content  
                  
                find_pg = find_page_number(s2.page_content)  
                if find_pg is not None:  
                    pg_number = find_pg  
                  
                sub_section_content += s2.page_content  
                json_data = {  
                    "doc_id": f"{job_id}-{chunk_id}",  
                    "chunk_id": chunk_id,  
                    "pg_number": int(pg_number),  
                    "file_name": os.path.basename(job_request.url_file_to_process),  
                    "content": sub_section_content,  
                    "title": title,  
                    "vector": generate_embedding(sub_section_content, job_request.openai_embedding_api_version, job_request.openai_embedding_api_base, job_request.openai_embedding_api_key, job_request.openai_embedding_model)  
                }  
                chunk_id += 1  
                documents.append(json_data)  
        else:  
            title = ''  
            if 'Header 1' in s1.metadata:  
                title = '# ' + s1.metadata['Header 1']  
                section_content = '# ' + s1.metadata['Header 1'] + '\n' + section_content  
              
            find_pg = find_page_number(section_content)  
            if find_pg is not None:  
                pg_number = find_pg  
              
            json_data = {  
                "doc_id": f"{job_id}-{chunk_id}",  
                "chunk_id": chunk_id,  
                "pg_number": int(pg_number),  
                "file_name": os.path.basename(job_request.url_file_to_process),  
                "content": section_content,  
                "title": title,  
                "vector": generate_embedding(section_content, job_request.openai_embedding_api_version, job_request.openai_embedding_api_base, job_request.openai_embedding_api_key, job_request.openai_embedding_model)  
            }  
            chunk_id += 1  
            documents.append(json_data)  
      
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


def find_page_number(text):  
    # Regular expression to find ||integer||  
    pattern = r'\|\|(\d+)\|\|'  
      
    # Find all matches  
    matches = re.findall(pattern, text)  
      
    # If there are matches, return the last one as an integer  
    if matches:  
        return int(matches[-1])  
    else:  
        return None  
        
# Download a file based on a url
def download_file(url, local_filename):  
    # Send a GET request to the URL  
    response = requests.get(url, stream=True)  
    # Open the local file in write-binary mode  
    with open(local_filename, 'wb') as file:  
        # Write the response content to the file in chunks  
        for chunk in response.iter_content(chunk_size=8192):  
            if chunk:  # Filter out keep-alive new chunks  
                file.write(chunk)  
    return local_filename  

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
def convert_to_pdf(input_path, pdf_dir):  
    output_file = input_path.replace(pathlib.Path(input_path).suffix, '')
    output_file = os.path.join(pdf_dir, os.path.basename(output_file + '.pdf'))

    if os.path.exists(output_file):
        os.remove(output_file)
      
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
    return output_file

# Convert pages from PDF to images
def extract_pdf_pages_to_images(pdf_path, image_out_dir):
    # Validate image_out directory exists
    ensure_directory_exists(image_out_dir)  

    # Open the PDF file and iterate pages
    print ('Extracting images from PDF...')
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
    
def list_files_with_relative_paths(directory):  
    file_paths = []  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            # Get the relative path  
            relative_path = os.path.relpath(os.path.join(root, file), directory)  
            file_paths.append(relative_path)  
    return file_paths  

#@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def extract_markdown_from_image(gpt_client, openai_gpt_model, image_path, prompt):
    max_attempts = 6
    max_backoff = 60

    counter = 0
    incremental_backoff = 1   # seconds to wait on throttline - this will be incremental backoff
    try:
        base64_image = encode_image(image_path)
        print ('Length of Base64 Image:', len(base64_image))
        user_prompt = """Extract everything you see in this image to markdown. 
                        Convert all charts such as line, pie and bar charts to markdown tables and include a note that the numbers are approximate.
                    """
        if prompt != None:
            user_prompt = prompt
        
        while True and counter < max_attempts:
            response = gpt_client.chat.completions.create(
                model=openai_gpt_model,
                messages=[
                    { "role": "system", "content": "You are a helpful assistant." },
                    { "role": "user", "content": [  
                        { 
                            "type": "text", 
                            "text": user_prompt 
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ] } 
                ],
                max_tokens=4096
            )
            print ('Received Response')
            # print (response.choices)
            return response.choices[0].message.content
    except OpenAIError as ex:
        # Handlethrottling - code 429
        print ('API Error Found')
        if str(ex.code) == "429":
            print ('Waiting to retry after', incremental_backoff, 'seconds...')
            incremental_backoff = min(max_backoff, incremental_backoff * 1.5)
            counter += 1
            time.sleep(incremental_backoff)
        elif str(ex.code) == "content_filter":
            print ('Conten Filter Error', ex.code)
            return "Content could not be extracted due to Azure OpenAI content filter." + ex.code
        else:
            print ('API Error:', ex)
    except Exception as ex:
        counter += 1
        print ('Error - Retry count:', counter, ex)
        
    return ""
    

def process_image(markdown_out_dir, openai_gpt_api_key, openai_gpt_api_version, openai_gpt_api_base, openai_gpt_model, file, prompt):
  
    if '.png' in file:
        print ('Processing:', file)
        # print ('openai_gpt_api_key', openai_gpt_api_key)
        # print ('openai_gpt_api_version', openai_gpt_api_version)
        # print ('openai_gpt_api_base', openai_gpt_api_base)
        # print ('openai_gpt_model', openai_gpt_model)
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
        print ('Skipping non PNG file:', file)

    return file
  
def extract_numeric_value(filename):  
    # Extract numeric value from filename using regular expression  
    match = re.search(r'(\d+)', filename)  
    return int(match.group(1)) if match else float('inf') 

# Upload status
def upload_current_status(blob_container_client, job_dir, status, msg):
    # Initialize the BlobServiceClient using the connection string  
    status['message'] = msg
    blob_client = blob_container_client.get_blob_client(os.path.join(job_dir, 'status.json'))
    blob_client.upload_blob(json.dumps(status, indent=4), overwrite=True)  

    return status


# Upload processed files to Blob Storage
def upload_files_to_blob_storage(blob_container_client, container_name, job_dir):  
    # Iterate through the files in the directory  
    files = list_files_with_relative_paths(job_dir)
    for file_path in files: 
        try:  
            # Create a blob client  
            blob_client = blob_container_client.get_blob_client(os.path.join(job_dir, file_path))
              
            # Upload the file  
            with open(os.path.join(job_dir, file_path), "rb") as data:  
                blob_client.upload_blob(data, overwrite=True)  
            print(f"Uploaded {file_path} to container {container_name}.")  
        except Exception as e:  
            print(f"Failed to upload {file_path}: {e}")  
  

def ensure_container(blob_container_client):
    # Ensure the container exists (create if it doesn't exist)  
    try:  
        blob_container_client.create_container()  
        print(f"Container created")  
    except Exception as e:  
        print(f"Container exists")  

# Function to generate vectors for text
max_attempts = 6
max_backoff = 60
def generate_embedding(text, openai_embedding_api_version, openai_embedding_api_base, openai_embedding_api_key, openai_embeddings_model):
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
            print ('API Error Found')
            if str(ex.code) == "429":
                print ('Waiting to retry after', incremental_backoff, 'seconds...')
                incremental_backoff = min(max_backoff, incremental_backoff * 1.5)
                counter += 1
                time.sleep(incremental_backoff)
            elif str(ex.code) == "content_filter":
                print ('Conten Filter Error', ex.code)
                return None
            else:
                print ('API Error - Retry count:', counter, ex.code)
            counter += 1
        except Exception as ex:
            counter += 1
            print ('Error - Retry count:', counter, ex)
            
    return None

# Submit a job and return details to user to monitor progress via job_status
@app.post("/start-job")  
async def start_job(job_request: JobRequest, background_tasks: BackgroundTasks):  
    try:
        # Create a job ID from the url that is base64 encoded - for security remove the SAS part if there is one
        download_url = job_request.url_file_to_process
        if '?' in download_url:
            download_url = download_url[:download_url.find('?')]

        job_id = encode_base64(download_url)
        job_info = {"job_id": job_id}
        # Start the background task and return the job details
        background_tasks.add_task(background_task, job_id, job_request)  
        return job_info
    except Exception as ex:
        job_info['status'] = "error"
        print ('Error:', ex)
        job_info['status'] = "error"
        job_info = upload_current_status(blob_container_client, job_dir, job_info, ex)
    
# Get the status of the job
@app.post("/job-status")  
async def job_status(job_status: JobStatus):  
    # Create connection to Azure Instances
    job_id = job_status.job_id
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={job_status.blob_storage_service_name};AccountKey={job_status.blob_storage_service_api_key};EndpointSuffix=core.windows.net"  
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)  
    blob_name = os.path.join('processed', job_id, 'status.json')
    print (blob_name)
    blob_client = blob_service_client.get_blob_client(container=job_status.blob_storage_container, blob=blob_name)  
    job_info = json.loads(blob_client.download_blob().readall())
    return job_info 
