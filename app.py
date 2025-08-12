import streamlit as st
import pdfplumber
import re
from sentence_transformers import SentenceTransformer, util
from docx import Document
import nltk

nltk.download('punkt')

# Functions to extract and clean text
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text, convert_to_tensor=True)

def calculate_similarity(text1, text2):
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    cosine_sim = util.pytorch_cos_sim(emb1, emb2)
    return cosine_sim.item()

# Streamlit UI
st.title("Resume and Job Description Similarity Checker")

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
uploaded_jd = st.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_resume and uploaded_jd:
    try:
        # Extract text from resume PDF
        resume_text = extract_text_from_pdf(uploaded_resume)
        
        # Extract text from JD depending on file type
        if uploaded_jd.type == "application/pdf":
            jd_text = extract_text_from_pdf(uploaded_jd)
        else:
            jd_text = extract_text_from_docx(uploaded_jd)
        
        # Clean texts
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(jd_text)
        
        # Calculate similarity
        score = calculate_similarity(resume_clean, jd_clean)
        
        st.success(f"Resume & Job Description similarity score: {score:.4f}")
    except Exception as e:
        st.error(f"Error processing files: {e}")
