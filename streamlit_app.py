from pydoc import doc
import streamlit as st

st.title("How Does Your Resume Compare?")
st.write(
    "Upload your resume and the job description to see how your resume compares."
)

from docx import Document
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Function to extract text from files
def extract_text(file):
    if file.name.endswith('.docx'):
        doc = Document(file)
        return ' '.join([p.text for p in doc.paragraphs])
    elif file.name.endswith('.pdf'):
        reader = PdfReader(file)
        return ' '.join([page.extract_text() for page in reader.pages])
    else:
        st.error("Unsupported file format. Please upload .docx or .pdf files.")
        return None

# Scoring function
def score_resume(resume_text, job_description_text):
    texts = [resume_text, job_description_text]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return score[0][0] * 100  # Convert to percentage

# Extract keywords from text
def extract_keywords(text, top_n=10):
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = {word: words.count(word) for word in set(words)}
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [keyword for keyword, freq in sorted_keywords[:top_n]]

# Streamlit App
st.title("Resume Scoring AI")
st.subheader("Upload your resume and job description to get a score and actionable feedback!")

# File upload widgets
resume_file = st.file_uploader("Upload Resume (.docx or .pdf)", type=["docx", "pdf"])
job_description_file = st.file_uploader("Upload Job Description (.docx or .pdf)", type=["docx", "pdf"])

if st.button("Score Resume"):
    if resume_file and job_description_file:
        # Extract text
        resume_text = extract_text(resume_file)
        job_description_text = extract_text(job_description_file)

        if resume_text and job_description_text:
            # Calculate score
            score = score_resume(resume_text, job_description_text)
            st.success(f"Your resume scored: {score:.2f}%")

            # Feedback
            st.subheader("Feedback")
            resume_keywords = extract_keywords(resume_text, top_n=10)
            job_description_keywords = extract_keywords(job_description_text, top_n=10)

            st.write("**Top Keywords in Your Resume**:")
            st.write(', '.join(resume_keywords))

            st.write("**Top Keywords in the Job Description**:")
            st.write(', '.join(job_description_keywords))

            missing_keywords = set(job_description_keywords) - set(resume_keywords)
            if missing_keywords:
                st.warning("Consider adding the following keywords to your resume for better alignment:")
                st.write(', '.join(missing_keywords))
            else:
                st.success("Your resume covers most of the critical keywords!")
    else:
        st.error("Please upload both files before scoring.")
