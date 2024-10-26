from fastapi import FastAPI, UploadFile
from pymongo import MongoClient
import os
import PyPDF2
import math
from dotenv import load_dotenv

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

def extract_keywords(doc, num_keywords=5):
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
    
@app.get("/test-mongo")
def test_mongo():
    # Test fetching some data from MongoDB
    doc_count = collection.count_documents({})
    return {"status": "connected", "document_count": doc_count}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
