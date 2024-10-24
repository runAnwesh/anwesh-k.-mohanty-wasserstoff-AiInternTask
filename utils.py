import os
import PyPDF2
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import os

# MongoDB connection setup
client = MongoClient('mongodb://localhost:27017/')
db = client['pdf_summary']
collection = db['documents']

nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('punkt_tab')

# Summarization using basic sentence extraction
def summarize(text, max_sentences=5):
    sentences = nltk.sent_tokenize(text)
    return " ".join(sentences[:max_sentences])

# Keyword extraction using TF-IDF
def extract_keywords(text, num_keywords=5):
    tfidf = TfidfVectorizer(stop_words='english', max_features=num_keywords)
    tfidf_matrix = tfidf.fit_transform([text])
    keywords = tfidf.get_feature_names_out()
    return keywords

def process_document(doc):
    summary = summarize(doc["content"])
    keywords = extract_keywords(doc["content"])
    
    # Ensure the keywords are a plain Python list
    keywords = keywords.tolist() if hasattr(keywords, 'tolist') else keywords
    
    # Update the MongoDB document with summary and keywords
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"summary": summary, "keywords": keywords}}
    )


def parse_pdf(file_path):
    metadata = {
        "document_name": os.path.basename(file_path),
        "path": file_path,
        "size": os.path.getsize(file_path),
    }
    # Read the PDF content
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    
    metadata["content"] = text
    return metadata
