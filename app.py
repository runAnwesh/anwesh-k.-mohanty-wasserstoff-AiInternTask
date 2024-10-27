import streamlit as st
from pymongo import MongoClient
import PyPDF2
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from dotenv import load_dotenv
import io

nlp = spacy.load("en_core_web_sm")
nltk.download('punkt_tab')

# Load environment variables from .env file
load_dotenv()

# MongoDB setup
mongo_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongo_uri)
db = client['pdf_summary']
collection = db['documents']

# Custom functions
def parse_pdf(file):
    metadata = {
        "document_name": file.name,
        "size": file.size,
    }
    
    # Read PDF content
    pdf_content = file.read()
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
    text = ""
    for page in pdf_reader.pages[:5]:  # Limit to first 5 pages for testing
        text += page.extract_text() or ""
    
    metadata["content"] = text
    return metadata

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
    
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"summary": summary, "keywords": keywords}}
    )
    return summary, keywords

# Streamlit interface
st.title("PDF Summarizer and Keyword Extractor")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    st.write("Processing the PDF file...")
    metadata = parse_pdf(uploaded_file)
    doc_id = collection.insert_one(metadata).inserted_id

    # Process the document and display results
    summary, keywords = process_document(metadata)
    
    st.subheader("Summary")
    st.write(summary)
    
    st.subheader("Keywords")
    st.write(", ".join(keywords))
    
    st.success("PDF processed and stored in MongoDB!")
