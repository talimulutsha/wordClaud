import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import PyPDF2
from docx import Document
import base64
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re

# Download NLTK stopwords data
nltk.download('stopwords')

# Set up standard stopwords
STANDARD_STOPWORDS = set(stopwords.words('english'))

# Add custom stopwords if needed
CUSTOM_STOPWORDS = {
    'said', 'one', 'two', 'would', 'could', 'also', 'us', 'may', 
    'might', 'shall', 'must', 'upon', 'yet', 'however'
}
FINAL_STOPWORDS = STANDARD_STOPWORDS.union(CUSTOM_STOPWORDS)

# Function for text cleaning
def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# File reading functions
def read_txt(file):
    return clean_text(file.getvalue().decode("utf-8"))

def read_docx(file):
    doc = Document(file)
    return clean_text(" ".join([para.text for para in doc.paragraphs]))

def read_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = " ".join([page.extract_text() for page in pdf.pages])
    return clean_text(text)

# Function to filter stopwords
def filter_stopwords(text, additional_stopwords=None):
    if additional_stopwords:
        all_stopwords = FINAL_STOPWORDS.union(set(additional_stopwords))
    else:
        all_stopwords = FINAL_STOPWORDS
        
    words = text.split()
    filtered_text = " ".join([word for word in words if word not in all_stopwords])
    return filtered_text

# Function to create download link
def get_download_link(data, filename, label, file_type='csv'):
    if file_type == 'csv':
        content = data.to_csv(index=False)
        mime_type = 'text/csv'
    elif file_type == 'png':
        content = data
        mime_type = 'image/png'
    
    b64 = base64.b64encode(content).decode() if file_type == 'png' else base64.b64encode(content.encode()).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{label}</a>'
    return href

# Streamlit app configuration
st.set_page_config(
    page_title="World Claude - Text Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main app
st.title("📊 World Claude - Advanced Text Analysis")
st.markdown("""
    Upload text documents (TXT, PDF, DOCX) for comprehensive text analysis including:
    - Word frequency analysis
    - Interactive word clouds
    - Stopword filtering
    - Data export capabilities
""")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    remove_stopwords = st.checkbox("Remove stopwords", value=True)
    additional_stopwords = st.text_input("Add custom stopwords (comma separated)", "")
    
    st.header("📊 Visualization Settings")
    wordcloud_width = st.slider("Word Cloud Width", 400, 1200, 800)
    wordcloud_height = st.slider("Word Cloud Height", 200, 800, 400)
    max_words = st.slider("Max Words in Cloud", 50, 500, 200)
    
    st.header("ℹ️ About")
    st.markdown("""
    Created by: [Talumul Islam Utsha](https://github.com/talimulutsha)  
    Contact: [talimul.ds@gmail.com](mailto:talimul.ds@gmail.com)  
    LinkedIn: [Profile](https://www.linkedin.com/in/talimul-islam-utsha-05b74b286/)
    """)

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload your document (TXT, PDF, or DOCX)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    try:
        # Read file based on type
        if uploaded_file.name.endswith('.txt'):
            text = read_txt(uploaded_file)
        elif uploaded_file.name.endswith('.pdf'):
            text = read_pdf(uploaded_file)
        elif uploaded_file.name.endswith('.docx'):
            text = read_docx(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a .txt, .pdf, or .docx file.")
            st.stop()
        
        # Show basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Words", len(text.split()))
        with col2:
            st.metric("Unique Words", len(set(text.split())))
        with col3:
            st.metric("Characters", len(text))
        
        # Process stopwords
        custom_stopwords = [word.strip() for word in additional_stopwords.split(',')] if additional_stopwords else []
        
        if remove_stopwords:
            processed_text = filter_stopwords(text, custom_stopwords)
        else:
            processed_text = text
        
        # Display processed text
        with st.expander("🔍 View Processed Text"):
            st.text_area("Processed Text", processed_text, height=200)
        
        # Word frequency analysis
        st.subheader("📈 Word Frequency Analysis")
        word_counts = Counter(processed_text.split())
        word_df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['Count'])
        word_df = word_df.sort_values('Count', ascending=False)
        
        # Show top words
        top_n = st.slider("Show top N words", 10, 100, 20)
        st.bar_chart(word_df.head(top_n))
        
        # Full word frequency table
        with st.expander("📋 View Full Word Frequency Table"):
            st.dataframe(word_df)
            st.markdown(get_download_link(word_df, "word_frequencies.csv", "📥 Download as CSV"), unsafe_allow_html=True)
        
        # Word cloud generation
        st.subheader("☁️ Word Cloud Visualization")
        if st.checkbox("Generate Word Cloud", value=True):
            wordcloud = WordCloud(
                width=wordcloud_width,
                height=wordcloud_height,
                background_color='white',
                max_words=max_words,
                colormap='viridis'
            ).generate(processed_text)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
            # Download word cloud
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
            buf.seek(0)
            st.markdown(get_download_link(buf.getvalue(), "wordcloud.png", "📥 Download Word Cloud", 'png'), unsafe_allow_html=True)
            plt.close(fig)
        
        # N-gram analysis
        st.subheader("🔠 N-gram Analysis")
        n_gram_size = st.selectbox("Select n-gram size", [2, 3, 4], index=0)
        
        def generate_ngrams(text, n):
            words = text.split()
            ngrams = zip(*[words[i:] for i in range(n)])
            return [' '.join(ngram) for ngram in ngrams]
        
        ngrams = generate_ngrams(processed_text, n_gram_size)
        ngram_counts = Counter(ngrams)
        ngram_df = pd.DataFrame.from_dict(ngram_counts, orient='index', columns=['Count'])
        ngram_df = ngram_df.sort_values('Count', ascending=False)
        
        st.bar_chart(ngram_df.head(top_n))
        with st.expander("📋 View N-gram Table"):
            st.dataframe(ngram_df)
            st.markdown(get_download_link(ngram_df, f"{n_gram_size}_grams.csv", "📥 Download N-grams"), unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ An error occurred during processing: {str(e)}")
        st.stop()

else:
    st.info("ℹ️ Please upload a document to begin analysis")