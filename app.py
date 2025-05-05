import streamlit as st
import fitz  # PyMuPDF
from docx import Document
import pytesseract
from PIL import Image
import re
import joblib

# Load the pre-trained model and vectorizer
model = joblib.load('resume_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Global variables
resume_data = {"text": "", "cleaned": "", "vectorized": None, "filename": ""}

# Helper functions
def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text.lower()

def extract_resume_text(file, filename):
    try:
        if filename.endswith('.pdf'):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return "".join([page.get_text() for page in doc])
        elif filename.endswith('.docx'):
            doc = Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif filename.endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(file)
            return pytesseract.image_to_string(img)
        else:
            return ""
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def summarize_resume(text):
    lines = text.split('\n')
    keywords = ['education', 'experience', 'skills', 'project', 'internship', 'certification']
    summary_lines = [line.strip() for line in lines if any(k in line.lower() for k in keywords)]
    return "\n".join(summary_lines[:10]) if summary_lines else "Could not extract any meaningful summary."

def model_predict(text):
    # Clean and vectorize the resume text
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])

    # Predict the job category using the pre-trained model
    category = model.predict(vectorized_text)[0]
    return category

def score_resume(text):
    # Score the resume based on how well the keywords match the predicted category
    category = model_predict(text)
    category_keywords = {
        "Software Engineer": ["python", "java", "c++", "software", "api", "backend", "frontend"],
        "Data Scientist": ["python", "machine learning", "data", "model", "pandas", "numpy"],
        "Project Manager": ["project", "budget", "timeline", "agile", "scrum", "lead"]
    }

    keywords = category_keywords.get(category, [])
    matched_keywords = sum(1 for word in keywords if word in text.lower())
    score = (matched_keywords / len(keywords)) * 100 if keywords else 50
    return int(score)

def suggest_improvements(text):
    suggestions = []
    if "objective" not in text.lower():
        suggestions.append("Consider adding a career objective.")
    if "experience" not in text.lower():
        suggestions.append("Include your professional experience.")
    if "skills" not in text.lower():
        suggestions.append("Highlight your technical or soft skills.")
    if len(text.split()) < 100:
        suggestions.append("Add more details to strengthen your resume.")
    return suggestions or ["Resume looks good!"]

# --- Streamlit UI ---
st.set_page_config(page_title="🤖 Resume Assistant", layout="centered")
st.title("🤖 Resume Assistant")

# Custom Styling
st.markdown("""
    <style>
        .stButton > button {
            background-color: #0072B1;
            color: white;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #005f8d;
        }
        .stTextArea textarea {
            background-color: #f4f4f4;
        }
        .stAlert {
            background-color: #f0f8ff;
        }
    </style>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, or Image)", type=["pdf", "docx", "jpg", "jpeg", "png"])

if uploaded_file:
    filename = uploaded_file.name
    content = extract_resume_text(uploaded_file, filename)
    if not content.strip():
        st.error("Failed to extract any text. Try another file.")
    else:
        st.success(f"Resume uploaded: {filename}")
        resume_data["text"] = content
        resume_data["cleaned"] = clean_text(content)

        # Display extracted text
        st.subheader("📄 Extracted Resume Text")
        st.text_area("Text", value=resume_data["text"], height=200)

        # Options
        st.subheader("🔧 Resume Analysis")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Generate Summary", use_container_width=True):
                st.info(summarize_resume(resume_data["text"]))

            if st.button("🛠️ Suggest Improvements", use_container_width=True):
                improvements = suggest_improvements(resume_data["cleaned"])
                for suggestion in improvements:
                    st.warning(f"• {suggestion}")

        with col2:
            if st.button("🔍 Predict Job Category", use_container_width=True):
                category = model_predict(resume_data["text"])
                st.success(f"Predicted Job Category: {category}")

            if st.button("📊 Score Resume", use_container_width=True):
                score = score_resume(resume_data["text"])
                st.success(f"Resume Score: {score} / 100")

# Add a custom footer
st.markdown("""
    <style>
        .footer {
            font-size: 12px;
            color: grey;
            text-align: center;
            padding: 10px;
        }
    </style>
    <div class="footer">
        Developed by 🤖 Resume Assistant | GIKI AI
    </div>
""", unsafe_allow_html=True)
