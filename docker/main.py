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
import uuid 

# For LibreOffice Doc Conversion to PDF
import subprocess  
import pathlib

# Image extraction from PDF
import fitz  # PyMuPDF  
from pathlib import Path  

# Image processing via GPT-4o  
from openai import AzureOpenAI, OpenAIError
from tenacity import retry, wait_random_exponential, stop_after_attempt 
import io
import base64
# from IPython.display import Markdown, display  

# Upload processed files to Azure Blob Storage
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient  

supported_conversion_types = ['.pptx', '.ppt', '.docx', '.doc', '.xlsx', '.xls']


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
    openai_embeddings_api_base: Optional[str] = None
    openai_embeddings_api_key: Optional[str] = None
    openai_embeddings_api_version: Optional[str] = None
    openai_embeddings_model: Optional[str] = None
    test: Optional[str] = None

class JobStatus(BaseModel):  
    job_id: str
    blob_storage_service_name: str
    blob_storage_service_api_key: str
    blob_storage_container: str

def background_task(job_id: str, job_request: JobRequest):  
    try:
        # Create a job ID and JSON for the job info / status
        job_info = {"job_id": job_id}
        job_info['status'] = "in-progress"
        job_info['message'] = "Initiating job and validating services..."
        
        ensure_directory_exists(job_id)
        
        # Create connection to Azure Instances
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={job_request.blob_storage_service_name};AccountKey={job_request.blob_storage_service_api_key};EndpointSuffix=core.windows.net"  
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)  
        blob_container_client = blob_service_client.get_container_client(job_request.blob_storage_container)  
        
        # Ensure the container for uploading results exits, create if not
        ensure_container(blob_container_client)
        
        job_info = upload_current_status(blob_container_client, job_id, job_info, "Starting processing...")
        
        # Reset the output directories
        if os.path.exists('doc'):
            remove_directory('doc')
        if os.path.exists('json'):
            remove_directory('json')
        if os.path.exists('images'):
            remove_directory('images')
        if os.path.exists('markdown'):
            remove_directory('markdown')
        if os.path.exists('pdf'):
            remove_directory('pdf')

        job_info = upload_current_status(blob_container_client, job_id, job_info, "Downloading file for processing...")

        # Download file locally for processing
        ensure_directory_exists('doc')
        doc_dir = os.path.join('doc', job_id)
        ensure_directory_exists(doc_dir)
        # Remove SAS token if it was passed
        download_file_name = os.path.basename(job_request.url_file_to_process)
        if '?' in download_file_name:
            download_file_name = download_file_name[:download_file_name.find('?')]
        print ('File to be processed:', download_file_name)
        download_path = os.path.join(doc_dir, download_file_name)
        file_to_process = download_file(job_request.url_file_to_process, download_path)
        print(f"Downloaded {job_request.url_file_to_process} to {file_to_process}")  

        job_info = upload_current_status(blob_container_client, job_id, job_info, "Checking if file needs to be converted to PDF...")
        # If the file is not a PDF and can be converted - do so
        file_path = pathlib.Path(file_to_process).suffix.lower()

        # Ensure the output directory exists  
        pdf_dir = os.path.join(job_id, 'pdf')
        ensure_directory_exists(pdf_dir)

        if file_path in supported_conversion_types:
            print ('Converting file to PDF...')
            pdf_path = convert_to_pdf(file_to_process, pdf_dir)
        elif file_path == '.pdf':
            print ('No conversion needed for PDF...')
            shutil.copy(file_to_process, pdf_dir)  
            pdf_path = os.path.join(pdf_dir, os.path.basename(file_to_process))
            print("File copied from", file_to_process, 'to', pdf_dir)  
        else:
            job_info['status'] = "error"
            print ('Error:', 'File passed is not supported')
            job_info['message'] = 'File passed is not supported'
            return job_info
            
        if os.path.exists(pdf_path) == False:
            print ('Error:', 'File was not converted to PDF')
            job_info['status'] = "error"
            job_info['message'] = 'File was not converted to PDF'
            return job_info
        

        print ('PDF File to process:', pdf_path)

        job_info = upload_current_status(blob_container_client, job_id, job_info, "Converting PDF pages to images...")

        # Extract PDF pages to images
        image_dir = os.path.join(job_id, 'images')
        ensure_directory_exists(image_dir)

        pdf_images_dir = extract_pdf_pages_to_images(pdf_path, image_dir)
        print ('Images saved to:', pdf_images_dir)
        
        files = get_all_files(pdf_images_dir)  
        total_files = len(files)
        print ('Total Image Files to Process:', total_files)

        job_info = upload_current_status(blob_container_client, job_id, job_info, "Converting images to Markdown...")
        # Convert the images to markdown using GPT-4o 
        # Process pages in parallel - adjust worker count as needed
        max_workers = 10

        markdown_dir = os.path.join(job_id, 'markdown')
        ensure_directory_exists(markdown_dir)

        # Create a list of tuples containing the arguments to pass to the worker function  
        # tasks = [(1, 2), (3, 4), (5, 6)]
        tasks = []
        for file in files: 
            tasks.append((markdown_dir, job_request.openai_gpt_api_key, job_request.openai_gpt_api_version, job_request.openai_gpt_api_base, job_request.openai_gpt_model, file))
        
          
        # Using ThreadPoolExecutor with a limit of max_workers threads  
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  
            # Use a list comprehension to submit tasks to the executor  
            futures = [executor.submit(process_image, *task) for task in tasks]  
              
            # Iterate over the futures as they complete  
            for future in concurrent.futures.as_completed(futures):  
                result = future.result()  
                print(f'Result: {result}')  

        # Get all the markdown files and sort them by page number
        files = os.listdir(markdown_dir)  

        # Filter out non-txt files
        txt_files = [f for f in files if f.endswith('.txt')]  
            
        # Sort files based on numeric values extracted from filenames  
        sorted_files = sorted(txt_files, key=extract_numeric_value)  

        total_files = len(sorted_files)
        print ('Total Markdown Files:', total_files)
        
        job_info = upload_current_status(blob_container_client, job_id, job_info, "Uploading processed files to Azure Blob Storage...")
        
        # Upload files
        upload_files_to_blob_storage(blob_container_client, job_request.blob_storage_container, job_id)  
        # upload_files_to_blob_storage(blob_container_client, pdf_dir, job_request.blob_storage_container, job_id)  
        # upload_files_to_blob_storage(blob_container_client, markdown_dir, job_request.blob_storage_container, job_id)  
        
        remove_directory(job_id)
        
        print ('Processing complete.')
        job_info['status'] = 'complete'
        job_info = upload_current_status(blob_container_client, job_id, job_info, "Processing complete.")

    except Exception as ex:
        print (ex)
        job_info['status'] = 'error'
        job_info = upload_current_status(blob_container_client, job_id, job_info, "Error:" + ex)

   
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
def extract_markdown_from_image(gpt_client, openai_gpt_model, image_path):
    max_attempts = 6
    max_backoff = 60

    counter = 0
    incremental_backoff = 1   # seconds to wait on throttline - this will be incremental backoff
    try:
        base64_image = encode_image(image_path)
        print ('Length of Base64 Image:', len(base64_image))
        while True and counter < max_attempts:
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
                max_tokens=2048
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
    

def process_image(markdown_out_dir, openai_gpt_api_key, openai_gpt_api_version, openai_gpt_api_base, openai_gpt_model, file):
  
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

        markdown_text = extract_markdown_from_image(gpt_client, openai_gpt_model, file)
        markdown_file_out = os.path.join(markdown_out_dir, os.path.basename(file).replace('.png', '.txt'))
        print(markdown_file_out)
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
def upload_current_status(blob_container_client, job_id, status, msg):
    # Initialize the BlobServiceClient using the connection string  
    status['message'] = msg
    blob_client = blob_container_client.get_blob_client(os.path.join(job_id, 'status.json'))
    blob_client.upload_blob(json.dumps(status, indent=4), overwrite=True)  

    return status


# Upload processed files to Blob Storage
def upload_files_to_blob_storage(blob_container_client, container_name, job_id):  
    # Iterate through the files in the directory  
    files = list_files_with_relative_paths(job_id)
    for file_path in files: 
        try:  
            # Create a blob client  
            blob_client = blob_container_client.get_blob_client(os.path.join(job_id, file_path))
              
            # Upload the file  
            with open(os.path.join(job_id, file_path), "rb") as data:  
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


# Submit a job and return details to user to monitor progress via job_status
@app.post("/start-job")  
async def start_job(job_request: JobRequest, background_tasks: BackgroundTasks):  
    try:
        # Create a job ID and JSON for the job info / status
        job_id = str(uuid.uuid4())
        job_info = {"job_id": job_id}
        # Start the background task and return the job details
        background_tasks.add_task(background_task, job_id, job_request)  
        return job_info
    except Exception as ex:
        job_info['status'] = "error"
        print ('Error:', ex)
        job_info['status'] = "error"
        job_info = upload_current_status(blob_container_client, job_id, job_info, ex)
    
# Get the status of the job
@app.post("/job-status")  
async def job_status(job_status: JobStatus):  
    # Create connection to Azure Instances
    job_id = job_status.job_id
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={job_status.blob_storage_service_name};AccountKey={job_status.blob_storage_service_api_key};EndpointSuffix=core.windows.net"  
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)  
    blob_name = os.path.join(job_id, 'status.json')
    print (blob_name)
    blob_client = blob_service_client.get_blob_client(container=job_status.blob_storage_container, blob=blob_name)  
    job_info = json.loads(blob_client.download_blob().readall())
    return job_info 


#if __name__ == "__main__":  
#    import uvicorn  
#    uvicorn.run(app, host="0.0.0.0", port=3100)  
#    #uvicorn.run(app, host="0.0.0.0", port=3100, ssl_keyfile="key.pem", ssl_certfile="cert.pem")  
