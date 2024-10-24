import requests
import os
import json
from utils import parse_pdf, process_document, collection
from concurrent.futures import ThreadPoolExecutor

# Load dataset.json
with open('dataset.json') as f:
    dataset = json.load(f)

def download_pdf(url, save_path):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Download the PDF
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save the PDF file
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return save_path
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

# Ingest PDFs from URLs in dataset.json
def ingest_pdfs_from_dataset(dataset):
    for pdf_name, pdf_url in dataset.items():
        print(f"Processing {pdf_name} from {pdf_url}")
        save_path = f"./Dataset/{pdf_name}.pdf"
        download_pdf(pdf_url, save_path)
        if os.path.exists(save_path):
            metadata = parse_pdf(save_path)
            doc_id = collection.insert_one(metadata).inserted_id
            process_document(collection.find_one({"_id": doc_id}))

def process_pdfs_concurrently():
    docs = list(collection.find())
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_document, docs)

if __name__ == "__main__":
    try:
        ingest_pdfs_from_dataset(dataset)
        process_pdfs_concurrently()
    except Exception as e:
        print(f"Error processing PDFs: {e}")
