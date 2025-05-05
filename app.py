import streamlit as st
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Dummy functions (replace these with your actual implementations)
def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

def summarize_resume(text):
    return "This is a summary of the resume."

def score_resume(text, vectorizer):
    return 80  # Placeholder score

def suggest_improvements(text):
    return ["Use a stronger action verb in the experience section.", "Reduce redundant phrases."]

def model_predict(vectorized):
    return ["Software Engineer"]  # Placeholder category

# File handling and resume text extraction
def extract_resume_text(file, filename):
    try:
        if filename.endswith('.pdf'):
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
            return text
        elif filename.endswith('.docx'):
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        elif filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(file)
            text = pytesseract.image_to_string(img)
            return text
        else:
            return ""
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Streamlit app code
st.title("Resume Assistant")

file = st.file_uploader("Upload Resume (PDF, DOCX, or Image)", type=["pdf", "docx", "jpg", "png"])

if file:
    text = extract_resume_text(file, file.name)
    if text:
        cleaned_text = clean_text(text)
        vectorizer = TfidfVectorizer()
        vectorizer.fit([cleaned_text])
        vectorized = vectorizer.transform([cleaned_text])

        summary = summarize_resume(text)
        category = model_predict(vectorized)[0]
        score = f"{score_resume(cleaned_text, vectorizer)} / 100"
        improvements = "\n".join(suggest_improvements(cleaned_text))

        st.subheader("Resume Summary:")
        st.write(summary)

        st.subheader("Job Category:")
        st.write(category)

        st.subheader("Resume Score:")
        st.write(score)

        st.subheader("Suggestions for Improvement:")
        st.write(improvements)
    else:
        st.error("Failed to extract text from the resume.")
