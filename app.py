import streamlit as st
import pickle
import re
import nltk
import fitz  # PyMuPDF for PDFs
import docx  # For .docx files
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load saved model and vectorizer
model = pickle.load(open('log_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Extract text from uploaded file
def extract_text(file, file_type):
    if file_type == 'txt':
        return file.read().decode("utf-8")
    elif file_type == 'pdf':
        with fitz.open(stream=file.read(), filetype='pdf') as doc:
            return "\n".join(page.get_text() for page in doc)
    elif file_type == 'docx':
        doc = docx.Document(file)
        return "\n".join(para.text for para in doc.paragraphs)
    return ""

# Streamlit Interface
st.title("Resume Job Role Classifier")
st.markdown("Paste your resume text or upload a `.txt`, `.pdf`, or `.docx` file.")

# Text input box
text_input = st.text_area("Paste Resume Text Here", height=200)

# File uploader
uploaded_file = st.file_uploader("Or upload a resume file", type=["txt", "pdf", "docx"])

# Decide which input to use
resume_text = ""

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1]
    try:
        resume_text = extract_text(uploaded_file, file_type)
    except Exception as e:
        st.error(f"Could not read the file: {e}")

elif text_input.strip():
    resume_text = text_input

# Prediction Button
if st.button("Predict Job Role"):
    if not resume_text.strip():
        st.warning("Please provide resume content (either paste or upload).")
    else:
        cleaned = preprocess_text(resume_text)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)
        job_role = prediction[0]
        st.success(f"Predicted Job Role: **{job_role}**")
