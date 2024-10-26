from fastapi import FastAPI, UploadFile, File, background_tasks
from pymongo import MongoClient
import PyPDF2
import math
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# MongoDB setup
mongo_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongo_uri)
db = client['pdf_summary']
collection = db['documents']

def parse_pdf(file):
    metadata = {
        "document_name": file.filename,
        "size": file.file.tell(),
    }
    
    # Reset file pointer to the beginning
    file.file.seek(0)
    
    reader = PyPDF2.PdfReader(file.file)
    text = ""
    for page in reader.pages[:5]:  # Limit to first 5 pages for testing
        text += page.extract_text() or ""
    
    metadata["content"] = text
    return metadata

def process_document_in_background(doc_id):
    doc = collection.find_one({"_id": doc_id})
    summary = summarize(doc["content"])
    keywords = extract_keywords(doc["content"])
    
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"summary": summary, "keywords": keywords}}
    )

# Custom summarization and keyword extraction functions
def tokenize(text):
    sentence_endings = '.!?'
    sentences = []
    current_sentence = []
    for char in text:
        current_sentence.append(char)
        if char in sentence_endings:
            sentences.append(''.join(current_sentence).strip())
            current_sentence = []
    if current_sentence:  # Catch any remaining sentence
        sentences.append(''.join(current_sentence).strip())
    return sentences

def summarize(text, max_sentences=5):
    sentences = tokenize(text)
    return " ".join(sentences[:max_sentences])

def compute_tf(word_dict, doc):
    tf_dict = {}
    doc_count = len(doc)
    for word, count in word_dict.items():
        tf_dict[word] = count / doc_count
    return tf_dict

def compute_idf(doc_list):
    idf_dict = {}
    N = len(doc_list)
    
    # Count the number of documents containing each word
    for doc in doc_list:
        for word in doc:
            if word in idf_dict:
                idf_dict[word] += 1
            else:
                idf_dict[word] = 1
    
    for word, count in idf_dict.items():
        idf_dict[word] = math.log(N / float(count))
    
    return idf_dict

def compute_tf_idf(tf, idf):
    tf_idf = {}
    for word, tf_val in tf.items():
        tf_idf[word] = tf_val * idf[word]
    return tf_idf

def extract_keywords(doc, num_keywords=3):
    word_dict = {}
    doc_words = doc.split()
    
    # Word count
    for word in doc_words:
        word_dict[word] = word_dict.get(word, 0) + 1

    # Compute TF-IDF
    tf = compute_tf(word_dict, doc_words)
    idf = compute_idf([doc_words])  # For simplicity, we're using only one doc context
    tf_idf = compute_tf_idf(tf, idf)

    # Sort and get top keywords
    sorted_keywords = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_keywords[:num_keywords]]

def process_document(doc):
    summary = summarize(doc["content"])
    keywords = extract_keywords(doc["content"])
    
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"summary": summary, "keywords": keywords}}
    )

# Route to upload PDF and process it
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Parse PDF content and metadata directly from the file
        metadata = parse_pdf(file)
        doc_id = collection.insert_one(metadata).inserted_id

        # Run document processing in the background
        background_tasks.add_task(process_document_in_background, doc_id)
        
        return {"message": "PDF processed and stored", "document_id": str(doc_id)}
    
    except Exception as e:
        return {"error": str(e)}
