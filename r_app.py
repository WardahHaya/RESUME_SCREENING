import streamlit as st
import fitz  # PyMuPDF
from docx import Document
import pytesseract
from PIL import Image
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Set Streamlit config
st.set_page_config(page_title="💼 Resume Assistant", layout="wide")

# Style
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .title {
        color: #003366;
        font-size: 40px;
        font-weight: bold;
    }
    .button-style > button {
        background-color: #0077b6;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        margin: 10px;
    }
    .button-style > button:hover {
        background-color: #023e8a;
    }
    </style>
""", unsafe_allow_html=True)

# Session Initialization
if "users" not in st.session_state:
    st.session_state.users = {}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None

if "page" not in st.session_state:
    st.session_state.page = "login"

if "resume_data" not in st.session_state:
    st.session_state.resume_data = {"text": "", "cleaned": "", "vectorized": None, "filename": ""}

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = TfidfVectorizer()


# Helper functions
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
    text = st.session_state.resume_data["cleaned"]
    if any(word in text for word in ["machine", "data", "analysis"]):
        return "Data Scientist"
    elif any(word in text for word in ["project", "lead", "timeline"]):
        return "Project Manager"
    else:
        return "Software Engineer"

def score_resume(text, vectorizer):
    category_keywords = {
        "Software Engineer": ["python", "java", "c++", "software", "api", "backend", "frontend"],
        "Data Scientist": ["python", "machine learning", "data", "model", "pandas", "numpy"],
        "Project Manager": ["project", "budget", "timeline", "agile", "scrum", "lead"]
    }
    category = model_predict(vectorizer.transform([text]))
    keywords = category_keywords.get(category, [])
    matched = sum(1 for word in keywords if word in text.lower())
    score = (matched / len(keywords)) * 100 if keywords else 50
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

# Login / Sign-up Page
def login_page():
    st.title("🔐 Login to Resume Assistant")
    action = st.radio("Choose Action", ["Login", "Sign Up"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Submit"):
        if action == "Login":
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.session_state.logged_in_user = username
                st.session_state.page = "home"
                st.success(f"Welcome back, {username}!")
            else:
                st.error("Invalid username or password.")
        else:
            if username in st.session_state.users:
                st.error("Username already exists.")
            else:
                st.session_state.users[username] = password
                st.session_state.logged_in = True
                st.session_state.logged_in_user = username
                st.session_state.page = "home"
                st.success("Account created!")

# Home Page
def home_page():
    st.markdown("<div class='title'>💼 Resume Assistant</div>", unsafe_allow_html=True)
    st.markdown(f"👋 Hello, **{st.session_state.logged_in_user}**!")
    
    if st.button("🚪 Log Out"):
        st.session_state.logged_in = False
        st.session_state.logged_in_user = None
        st.session_state.page = "login"
        return

    uploaded_file = st.file_uploader("📁 Upload Resume (PDF, DOCX, or Image)", type=["pdf", "docx", "jpg", "jpeg", "png"])

    if uploaded_file:
        filename = uploaded_file.name
        content = extract_resume_text(uploaded_file, filename)
        if not content.strip():
            st.error("Failed to extract any text. Try another file.")
        else:
            st.session_state.resume_data["text"] = content
            st.session_state.resume_data["cleaned"] = clean_text(content)
            if not hasattr(st.session_state.vectorizer, 'vocabulary_'):
                st.session_state.vectorizer.fit([st.session_state.resume_data["cleaned"]])
            st.session_state.resume_data["vectorized"] = st.session_state.vectorizer.transform([st.session_state.resume_data["cleaned"]])

            st.markdown("### Choose an Action:")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Generate Summary"):
                    st.session_state.page = "summary"
            with col2:
                if st.button("🛠 Suggest Improvements"):
                    st.session_state.page = "improve"

            col3, col4 = st.columns(2)
            with col3:
                if st.button("🔍 Predict Job Category"):
                    st.session_state.page = "predict"
            with col4:
                if st.button("📊 Score Resume"):
                    st.session_state.page = "score"

# Feature Pages
def feature_page(title, body_fn):
    st.title(title)
    st.markdown(f"👤 Logged in as: **{st.session_state.logged_in_user}**")
    st.text_area("Resume Content", value=st.session_state.resume_data["text"], height=200)
    body_fn()
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"

def summary_body():
    summary = summarize_resume(st.session_state.resume_data["text"])
    st.info(summary)

def improvement_body():
    suggestions = suggest_improvements(st.session_state.resume_data["cleaned"])
    for s in suggestions:
        st.warning(f"• {s}")

def predict_body():
    category = model_predict(st.session_state.resume_data["vectorized"])
    st.success(f"Predicted Job Category: {category}")

def score_body():
    score = score_resume(st.session_state.resume_data["cleaned"], st.session_state.vectorizer)
    st.success(f"Resume Score: {score} / 100")

# Navigation
if not st.session_state.logged_in:
    login_page()
elif st.session_state.page == "home":
    home_page()
elif st.session_state.page == "summary":
    feature_page("📄 Resume Summary", summary_body)
elif st.session_state.page == "improve":
    feature_page("🛠 Resume Improvements", improvement_body)
elif st.session_state.page == "predict":
    feature_page("🔍 Job Category Prediction", predict_body)
elif st.session_state.page == "score":
    feature_page("📊 Resume Scoring", score_body)
