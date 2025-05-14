# RESUME_SCREENING Smart Resume Screener

An intelligent resume screening web application that leverages Natural Language Processing (NLP) to analyze, categorize, and improve resumes. Built with Streamlit and integrated with BERT using MindSpore, this tool streamlines the candidate evaluation process for recruiters and offers constructive feedback for job seekers.

---

## 🔍 Features

- 🔐 Secure login system with personalized sessions  
- 📄 Upload support for PDF and DOCX resume files  
- 🧠 TF-IDF vectorization and rule-based keyword matching  
- 🗂️ Resume classification into job categories  
- 📊 Resume scoring based on relevance  
- 🤖 Semantic analysis with BERT via MindSpore  
- 📝 Resume summarization and improvement suggestions  

---

## 🛠️ Tech Stack

**Frontend:**  
- [Streamlit](https://streamlit.io/)

**Backend & NLP:**  
- Python  
- Scikit-learn  
- PyMuPDF (`fitz`)  
- `python-docx`  
- Regular Expressions (`re`)  
- TF-IDF for feature extraction  
- Rule-based classification logic  
- BERT integration using [MindSpore](https://www.mindspore.cn/)

**Deployment:**  
- Streamlit Cloud  

### Prerequisites

Make sure you have the following installed:

- Python 3.8 or higher
- pip

**Clone the repository**

```bash
git clone https://github.com/your-username/smart-resume-screener.git
cd smart-resume-screener
