import streamlit as st
import fitz  # PyMuPDF
from docx import Document
import pytesseract
from PIL import Image
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Global variables
resume_data = {"text": "", "cleaned": "", "vectorized": None, "filename": ""}
vectorizer = TfidfVectorizer()

# Helper functions (same as before)
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

def model_predict(vectorized):
    text = resume_data["cleaned"]
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
st.set_page_config(page_title="💼 Resume Assistant", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Colors and Layout
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        font-size: 16px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextArea>textarea {
        background-color: #f0f8ff;
        color: #333;
    }
    .stAlert {
        background-color: #ffcccc;
        color: #e60000;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
    }
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
    }
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f4f4f9;
    }
    </style>
""", unsafe_allow_html=True)

# Login/Sign Up system
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    option = st.sidebar.selectbox("Choose an option", ["Login", "Sign Up"])

    if option == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            # Check login info here, for now using static values
            if username == "user" and password == "password":
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.session_state.page = "Home"  # Set default page after login
            else:
                st.error("Invalid credentials")

    elif option == "Sign Up":
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            st.session_state.logged_in = True
            st.session_state.username = new_username
            st.success(f"Account created for {new_username}!")
            st.session_state.page = "Home"  # Set default page after sign-up

else:
    # Logged In
    st.sidebar.title(f"Welcome {st.session_state.username}")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Generate Summary", "Suggest Improvements", "Predict Job Category", "Score Resume"], index=["Home", "Generate Summary", "Suggest Improvements", "Predict Job Category", "Score Resume"].index(st.session_state.page))

    uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, or Image)", type=["pdf", "docx", "jpg", "jpeg", "png"])

    if uploaded_file:
        filename = uploaded_file.name
        content = extract_resume_text(uploaded_file, filename)
        if not content.strip():
            st.error("Failed to extract any text. Try another file.")
        else:
            resume_data["text"] = content
            resume_data["cleaned"] = clean_text(content)

            if not hasattr(vectorizer, 'vocabulary_'):
                vectorizer.fit([resume_data["cleaned"]])
            resume_data["vectorized"] = vectorizer.transform([resume_data["cleaned"]])

            # Page-wise Content
            if page == "Home":
                st.title("Welcome to the Resume Assistant")
                st.markdown("""
                    This tool helps you analyze and improve your resume. Upload your resume file (PDF, DOCX, or image format) and explore the following features:
                    - **Generate Resume Summary**
                    - **Suggest Resume Improvements**
                    - **Predict Job Category**
                    - **Score Resume Based on Keywords**
                """)

            elif page == "Generate Summary":
                st.title("📄 Generate Resume Summary")
                st.text_area("Extracted Text", value=resume_data["text"], height=200)
                summary = summarize_resume(resume_data["text"])
                st.info(summary)
                if st.button("Back to Home"):
                    st.session_state.page = "Home"

            elif page == "Suggest Improvements":
                st.title("🛠 Suggest Improvements")
                st.text_area("Extracted Text", value=resume_data["text"], height=200)
                improvements = suggest_improvements(resume_data["cleaned"])
                for suggestion in improvements:
                    st.warning(f"• {suggestion}")
                if st.button("Back to Home"):
                    st.session_state.page = "Home"

            elif page == "Predict Job Category":
                st.title("🔍 Predict Job Category")
                st.text_area("Extracted Text", value=resume_data["text"], height=200)
                category = model_predict(resume_data["vectorized"])
                st.success(f"Predicted Job Category: {category}")
                if st.button("Back to Home"):
                    st.session_state.page = "Home"

            elif page == "Score Resume":
                st.title("📊 Score Resume")
                st.text_area("Extracted Text", value=resume_data["text"], height=200)
                score = score_resume(resume_data["cleaned"], vectorizer)
                st.success(f"Resume Score: {score} / 100")
                if st.button("Back to Home"):
                    st.session_state.page = "Home"

    if st.button("Log Out"):
        st.session_state.logged_in = False
        st.session_state.page = "Home"  # Reset page after logout
        st.experimental_rerun()
