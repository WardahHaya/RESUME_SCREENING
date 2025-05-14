import streamlit as st
import fitz  # PyMuPDF
from docx import Document
import pytesseract
from PIL import Image
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Simulated database for users
if 'users' not in st.session_state:
    st.session_state['users'] = {}

# Session State for authentication and flow
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'
if 'resume_uploaded' not in st.session_state:
    st.session_state['resume_uploaded'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

# Resume data state
if 'resume_data' not in st.session_state:
    st.session_state['resume_data'] = {"text": "", "cleaned": "", "vectorized": None, "filename": ""}

vectorizer = TfidfVectorizer()

# ------------------- Helper Functions -------------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()

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

def model_predict(vectorized):
    text = st.session_state['resume_data']['cleaned']
    if any(word in text for word in ["machine", "data", "analysis"]):
        return "Data Scientist"
    elif any(word in text for word in ["project", "lead", "timeline"]):
        return "Project Manager"
    else:
        return "Software Engineer"

def score_resume(text):
    category_keywords = {
        "Software Engineer": ["python", "java", "c++", "software", "api", "backend", "frontend"],
        "Data Scientist": ["python", "machine learning", "data", "model", "pandas", "numpy"],
        "Project Manager": ["project", "budget", "timeline", "agile", "scrum", "lead"]
    }
    category = model_predict(vectorizer.transform([text]))
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

def logout():
    st.session_state['authenticated'] = False
    st.session_state['page'] = 'login'
    st.session_state['resume_uploaded'] = False
    st.session_state['resume_data'] = {"text": "", "cleaned": "", "vectorized": None, "filename": ""}

# ------------------- Styling -------------------
st.markdown("""
    <style>
        body {
            background-color: #e0f7fa;
            font-family: 'Arial', sans-serif;
        }
        .main, .block-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3, h4 {
            color: #004d40;
        }
        .stButton>button {
            background-color: #00695c;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .stTextArea textarea {
            background-color: #f1f1f1;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------- Pages -------------------
if st.session_state['page'] == 'login':
    st.title("Resume Assistant")
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.session_state['users'] and st.session_state['users'][username] == password:
            st.session_state['authenticated'] = True
            st.session_state['username'] = username
            st.session_state['page'] = 'upload'
        else:
            st.error("Invalid username or password.")
    if st.button("Sign Up"):
        st.session_state['page'] = 'signup'

elif st.session_state['page'] == 'signup':
    st.title("Create Account")
    new_username = st.text_input("Choose Username")
    new_password = st.text_input("Choose Password", type="password")
    if st.button("Create Account"):
        if new_username in st.session_state['users']:
            st.warning("Username already exists.")
        else:
            st.session_state['users'][new_username] = new_password
            st.session_state['authenticated'] = True
            st.session_state['username'] = new_username
            st.session_state['page'] = 'upload'

elif st.session_state['authenticated'] and st.session_state['page'] == 'upload':
    st.title("Upload Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, or Image)", type=["pdf", "docx", "jpg", "jpeg", "png"])
    if uploaded_file:
        filename = uploaded_file.name
        content = extract_resume_text(uploaded_file, filename)
        if not content.strip():
            st.error("Failed to extract any text. Try another file.")
        else:
            st.session_state['resume_data']['text'] = content
            st.session_state['resume_data']['cleaned'] = clean_text(content)
            if not hasattr(vectorizer, 'vocabulary_'):
                vectorizer.fit([st.session_state['resume_data']['cleaned']])
            st.session_state['resume_data']['vectorized'] = vectorizer.transform([st.session_state['resume_data']['cleaned']])
            st.session_state['resume_uploaded'] = True
            st.session_state['page'] = 'features'

elif st.session_state['authenticated'] and st.session_state['page'] == 'features':
    st.title("Resume Analysis Options")
    st.button("Logout", on_click=logout)
    if st.button("Generate Summary"):
        st.session_state['page'] = 'summary'
    if st.button("Suggest Improvements"):
        st.session_state['page'] = 'improvements'
    if st.button("Predict Job Category"):
        st.session_state['page'] = 'prediction'
    if st.button("Score Resume"):
        st.session_state['page'] = 'score'

elif st.session_state['page'] == 'summary':
    st.title("Resume Summary")
    st.text_area("Resume Text", value=st.session_state['resume_data']['text'], height=200)
    st.info(summarize_resume(st.session_state['resume_data']['text']))
    if st.button("Back to Features"):
        st.session_state['page'] = 'features'

elif st.session_state['page'] == 'improvements':
    st.title("Resume Improvement Suggestions")
    st.text_area("Resume Text", value=st.session_state['resume_data']['text'], height=200)
    for suggestion in suggest_improvements(st.session_state['resume_data']['cleaned']):
        st.warning(f"- {suggestion}")
    if st.button("Back to Features"):
        st.session_state['page'] = 'features'

elif st.session_state['page'] == 'prediction':
    st.title("Predicted Job Category")
    st.text_area("Resume Text", value=st.session_state['resume_data']['text'], height=200)
    st.success(f"Predicted Category: {model_predict(st.session_state['resume_data']['vectorized'])}")
    if st.button("Back to Features"):
        st.session_state['page'] = 'features'

elif st.session_state['page'] == 'score':
    st.title("Resume Score")
    st.text_area("Resume Text", value=st.session_state['resume_data']['text'], height=200)
    st.success(f"Resume Score: {score_resume(st.session_state['resume_data']['cleaned'])} / 100")
    if st.button("Back to Features"):
        st.session_state['page'] = 'features'  

else:
    st.warning("Something went wrong. Please log in again.")
    logout()
