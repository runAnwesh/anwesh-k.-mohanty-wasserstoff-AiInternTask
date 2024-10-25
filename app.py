from fastapi import FastAPI, UploadFile
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
import os
import PyPDF2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import spacy
import shutil

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# MongoDB setup
mongo_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongo_uri)
db = client['pdf_summary']
collection = db['documents']

nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')

# Helper functions for PDF processing
def parse_pdf(file_path):
    metadata = {
        "document_name": os.path.basename(file_path),
        "path": file_path,
        "size": os.path.getsize(file_path),
    }
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    
    metadata["content"] = text
    return metadata

def summarize(text, max_sentences=5):
    sentences = nltk.sent_tokenize(text)
    return " ".join(sentences[:max_sentences])

def extract_keywords(text, num_keywords=5):
    tfidf = TfidfVectorizer(stop_words='english', max_features=num_keywords)
    tfidf_matrix = tfidf.fit_transform([text])
    keywords = tfidf.get_feature_names_out()
    return keywords

def process_document(doc):
    summary = summarize(doc["content"])
    keywords = extract_keywords(doc["content"])
    
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"summary": summary, "keywords": keywords}}
    )

# Route to upload PDF and trigger the pipeline
@app.post("/upload/")
async def upload_pdf(file: UploadFile):
    try:
        file_location = f"temp_pdfs/{file.filename}"
        with open(file_location, "wb+") as f:
            shutil.copyfileobj(file.file, f)
        
        metadata = parse_pdf(file_location)
        doc_id = collection.insert_one(metadata).inserted_id
        process_document(collection.find_one({"_id": doc_id}))
        
        return {"message": "PDF processed and stored", "document_id": str(doc_id)}
    except Exception as e:
        return {"error": str(e)}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
