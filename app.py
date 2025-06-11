# you need to install all these in your terminal
# pip install streamlit
# pip install scikit-learn
# pip install python-docx
# pip install PyPDF2


import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Load pre-trained model and TF-IDF vectorizer (ensure these are saved earlier)
svc_model = pickle.load(open('clf.pkl', 'rb'))  # Example file name, adjust as needed
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Example file name, adjust as needed
le = pickle.load(open('encoder.pkl', 'rb'))  # Example file name, adjust as needed


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    # Try using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the category of a resume
def pred(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = cleanResume(input_resume)

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = svc_model.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]  # Return the category name


# Function to extract keywords from text
def extract_keywords(text, top_n=10):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])
    
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get TF-IDF scores
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    # Create a list of (word, score) tuples
    word_scores = list(zip(feature_names, tfidf_scores))
    
    # Sort by score in descending order
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N keywords
    return [word for word, score in word_scores[:top_n]]


# Function to calculate match score and keywords
def analyze_job_match(resume_text, job_description):
    # Clean both texts
    clean_resume = cleanResume(resume_text)
    clean_job = cleanResume(job_description)
    
    # Extract keywords from both
    resume_keywords = set(extract_keywords(clean_resume))
    job_keywords = set(extract_keywords(clean_job))
    
    # Find matched and missing keywords
    matched_keywords = resume_keywords.intersection(job_keywords)
    missing_keywords = job_keywords - resume_keywords
    
    # Calculate match score using cosine similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_job])
    match_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return {
        'match_score': match_score,
        'matched_keywords': list(matched_keywords),
        'missing_keywords': list(missing_keywords)
    }


# Streamlit app layout
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Resume Analyzer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Theme toggle button in top right
    col1, col2 = st.columns([0.95, 0.05])
    with col2:
        if st.button("☀️" if st.session_state.theme == 'light' else "🌙", key="theme_toggle"):
            toggle_theme()
            st.rerun()

    # Custom CSS based on theme
    if st.session_state.theme == 'light':
        st.markdown("""
            <style>
            /* Light theme styles */
            .main {
                padding: 2rem;
                background-color: #ffffff;
            }
            .section {
                padding: 1.5rem 0;
                border-bottom: 1px solid #eef2f7;
            }
            .section:last-child {
                border-bottom: none;
            }
            .footer {
                text-align: center;
                color: #7f8c8d;
                margin-top: 2rem;
                padding: 1rem;
                border-top: 1px solid #eef2f7;
                font-size: 1.2rem;
                font-weight: 500;
            }
            .header-text {
                color: #2c3e50;
            }
            .subheader-text {
                color: #7f8c8d;
            }
            .section-header {
                color: #2c3e50;
                text-align: center;
            }
            /* Override Streamlit's default colors */
            .stApp {
                background-color: #ffffff;
            }
            .stTextArea textarea {
                background-color: #f0f2f6; /* Slightly darker background for text area */
                color: #2c3e50;
                border-color: #eef2f7;
                border-radius: 10px;
                padding: 1rem;
            }
            .stFileUploader {
                background-color: #e6f3ff;
                color: #2c3e50;
                border-radius: 10px;
                padding: 1rem;
                border: 2px dashed #eef2f7;
            }
            .stMarkdown {
                color: #2c3e50;
            }
            /* Make warning text visible in light theme */
            .stAlert p {
                color: #2c3e50 !important;
            }
            /* Expander styling for visibility in light theme, matching warning background */
            .stExpander {
                background-color: #e6f3ff; /* Matching the warning/drag-drop background */
                border: 1px solid #d3d9e0; /* Stronger border */
                border-radius: 10px;
                padding: 0.5rem 1rem;
                margin-top: 1rem;
            }
            .stExpander > div > div > p,
            .stExpander .stMarkdown p {
                color: #2c3e50 !important; /* Ensure all content text within expander is visible */
            }
            /* Ensure expander header text and arrow are always visible and not white/red on hover */
            .stExpander .streamlit-expanderHeader,
            .stExpander .streamlit-expanderHeader > div:first-child,
            .stExpander .streamlit-expanderHeader .st-emotion-cache-s1mvs1 p {
                color: #2c3e50 !important; /* Ensure header text and arrow are dark */
            }
            .stExpander .streamlit-expanderHeader:hover,
            .stExpander .streamlit-expanderHeader:hover > div:first-child,
            .stExpander .streamlit-expanderHeader:hover .st-emotion-cache-s1mvs1 p {
                color: #2c3e50 !important; /* Prevent color change on hover */
            }
            /* Button styling */
            .stButton button {
                border-radius: 50%;
                width: 40px;
                height: 40px;
                padding: 0;
                font-size: 1.2rem;
            }
            /* Category badge styling */
            .category-badge-wrapper {
                display: flex;
                justify-content: center;
                width: 100%;
                margin: 0.5rem 0;
            }
            .category-badge {
                display: inline-block;
                padding: 0.5rem 1rem;
                background: linear-gradient(135deg, #4CAF50, #45a049);
                color: white;
                border-radius: 20px;
                font-weight: 500;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            /* Match score styling */
            .match-score {
                font-size: 2rem;
                font-weight: bold;
                text-align: center;
                margin: 1rem 0;
                background: linear-gradient(135deg, #2196F3, #1976D2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            /* Analysis section styling */
            .analysis-section {
                text-align: center;
                margin: 1rem 0;
            }
            .analysis-header {
                color: #2c3e50;
                margin-bottom: 1rem;
                text-align: center;
                width: 100%;
            }
            /* Keyword list styling */
            .keyword-list {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem auto; /* Centers the block itself */
                display: flex; /* Arranges items inside */
                flex-wrap: wrap;
                justify-content: center; /* Centers items within the flex container */
                width: fit-content; /* Ensure the block shrinks to content width for centering */
            }
            .keyword-list ul {
                list-style-position: inside;
                display: inline-block;
                text-align: left;
                margin: 0;
                padding: 0;
            }
            .keyword-list li {
                margin: 0.3rem 0;
                color: #2c3e50;
            }
            /* Keyword button styling */
            .keyword-button {
                display: inline-block;
                padding: 0.4rem 0.8rem;
                margin: 0.3rem;
                border-radius: 15px;
                font-weight: 400;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                white-space: nowrap; /* Prevent keywords from wrapping */
            }
            .matched-keyword {
                background: linear-gradient(135deg, #4CAF50, #45a049);
                color: white;
            }
            .missing-keyword {
                background: linear-gradient(135deg, #FF6347, #E04B2F);
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            /* Dark theme styles */
            .main {
                padding: 2rem;
                background-color: #1a1a1a;
            }
            .section {
                padding: 1.5rem 0;
                border-bottom: 1px solid #2d2d2d;
            }
            .section:last-child {
                border-bottom: none;
            }
            .footer {
                text-align: center;
                color: #a0a0a0;
                margin-top: 2rem;
                padding: 1rem;
                border-top: 1px solid #2d2d2d;
                font-size: 1.2rem;
                font-weight: 500;
            }
            .header-text {
                color: #ffffff;
            }
            .subheader-text {
                color: #a0a0a0;
            }
            .section-header {
                color: #ffffff;
                text-align: center;
            }
            /* Override Streamlit's default dark theme colors */
            .stApp {
                background-color: #1a1a1a;
            }
            .stTextArea textarea {
                background-color: #2d2d2d;
                color: #ffffff;
                border-color: #3d3d3d;
                border-radius: 10px;
                padding: 1rem;
            }
            .stFileUploader {
                background-color: #2d2d2d;
                color: #ffffff;
                border-radius: 10px;
                padding: 1rem;
                border: 2px dashed #3d3d3d;
            }
            .stMarkdown {
                color: #ffffff;
            }
            /* Ensure expander icon and header text are always visible and not red on hover */
            .stExpander .streamlit-expanderHeader {
                color: #ffffff !important;
            }
            .stExpander .streamlit-expanderHeader > div:first-child {
                color: #ffffff !important; /* Targeting the arrow specifically */
            }
            .stExpander .streamlit-expanderHeader:hover {
                color: #ffffff !important; /* Prevent color change on hover */
            }
            .stExpander .streamlit-expanderHeader:hover > div:first-child {
                color: #ffffff !important; /* Prevent arrow color change on hover */
            }
            /* Button styling */
            .stButton button {
                border-radius: 50%;
                width: 40px;
                height: 40px;
                padding: 0;
                font-size: 1.2rem;
                background-color: #2d2d2d;
                color: #ffffff;
            }
            /* Category badge styling */
            .category-badge-wrapper {
                display: flex;
                justify-content: center;
                width: 100%;
                margin: 0.5rem 0;
            }
            .category-badge {
                display: inline-block;
                padding: 0.5rem 1rem;
                background: linear-gradient(135deg, #4CAF50, #45a049);
                color: white;
                border-radius: 20px;
                font-weight: 500;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            /* Match score styling */
            .match-score {
                font-size: 2rem;
                font-weight: bold;
                text-align: center;
                margin: 1rem 0;
                background: linear-gradient(135deg, #2196F3, #1976D2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            /* Analysis section styling */
            .analysis-section {
                text-align: center;
                margin: 1rem 0;
            }
            .analysis-header {
                color: #ffffff;
                margin-bottom: 1rem;
                text-align: center;
                width: 100%;
            }
            /* Keyword list styling for dark theme */
            .keyword-list {
                background-color: #2d2d2d;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem auto; /* Centers the block itself */
                display: flex; /* Arranges items inside */
                flex-wrap: wrap;
                justify-content: center; /* Centers items within the flex container */
                width: fit-content; /* Ensure the block shrinks to content width for centering */
            }
            .keyword-list ul {
                list-style-position: inside;
                display: inline-block;
                text-align: left;
                margin: 0;
                padding: 0;
            }
            .keyword-list li {
                margin: 0.3rem 0;
                color: #ffffff;
            }
            /* Keyword button styling for dark theme */
            .keyword-button {
                display: inline-block;
                padding: 0.4rem 0.8rem;
                margin: 0.3rem;
                border-radius: 15px;
                font-weight: 400;
                box-shadow: 0 1px 2px rgba(0,0,0,0.2);
                white-space: nowrap;
            }
            .matched-keyword {
                background: linear-gradient(135deg, #388E3C, #2C7A30);
                color: white;
            }
            .missing-keyword {
                background: linear-gradient(135deg, #C62828, #A91E1E);
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)

    # Header
    st.markdown("<h1 class='header-text' style='text-align: center;'>Resume Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader-text' style='text-align: center;'>Upload your resume and get instant analysis</p>", unsafe_allow_html=True)

    # Create three columns for the layout
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<h3 class='section-header'>📤 Upload Resume</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #7f8c8d; text-align: center;'>Supported formats: PDF, DOCX, TXT</p>", unsafe_allow_html=True)
        
        # File upload section
        uploaded_file = st.file_uploader("", type=["pdf", "docx", "txt"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<h3 class='section-header'>📝 Job Description</h3>", unsafe_allow_html=True)
        job_description = st.text_area("Paste the job description here", height=200)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<h3 class='section-header'>📊 Analysis Results</h3>", unsafe_allow_html=True)
        
        if uploaded_file is not None and job_description:
            try:
                # Show loading spinner
                with st.spinner('Analyzing your resume...'):
                    # Extract text from the uploaded file
                    resume_text = handle_file_upload(uploaded_file)
                    
                    # Make category prediction
                    category = pred(resume_text)
                    
                    # Analyze job match
                    match_analysis = analyze_job_match(resume_text, job_description)
                    
                    # Display category
                    st.markdown("<div class='analysis-section'>", unsafe_allow_html=True)
                    st.markdown("<h4 class='analysis-header'>Predicted Category</h4>", unsafe_allow_html=True)
                    st.markdown(f"<div class='category-badge-wrapper'><div class='category-badge'>{category}</div></div>", unsafe_allow_html=True)
                    
                    # Display match score
                    st.markdown("<h4 class='analysis-header'>Job Match Score</h4>", unsafe_allow_html=True)
                    match_percentage = match_analysis['match_score'] * 100
                    st.markdown(f"<div class='match-score'>{match_percentage:.1f}%</div>", unsafe_allow_html=True)
                    
                    # Display matched keywords
                    st.markdown("<h4 class='analysis-header'>Matched Keywords</h4>", unsafe_allow_html=True)
                    if match_analysis['matched_keywords']:
                        st.markdown("<div class='keyword-list'>", unsafe_allow_html=True)
                        for keyword in match_analysis['matched_keywords']:
                            st.markdown(f"<div class='keyword-button matched-keyword'>{keyword}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("No matching keywords found.")
                    
                    # Display missing keywords
                    st.markdown("<h4 class='analysis-header'>Missing Keywords</h4>", unsafe_allow_html=True)
                    if match_analysis['missing_keywords']:
                        st.markdown("<div class='keyword-list'>", unsafe_allow_html=True)
                        for keyword in match_analysis['missing_keywords']:
                            st.markdown(f"<div class='keyword-button missing-keyword'>{keyword}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.success("All keywords are present in your resume!")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Show extracted text in an expander
                    with st.expander("View Extracted Resume Text"):
                        st.text_area("", resume_text, height=300)
                    
                    st.success("Analysis completed successfully!")

            except Exception as e:
                st.error(f"Error processing the file: {str(e)}")
        else:
            if not uploaded_file:
                st.info("Please upload a resume to see the analysis results.")
            if not job_description:
                st.info("Please provide a job description to see the match analysis.")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("<div class='footer'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin: 0;'>Made with ❤️ by Bikramjit</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
