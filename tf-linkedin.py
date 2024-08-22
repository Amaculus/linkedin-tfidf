import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import Phrases, Word2Vec
from gensim.models.phrases import Phraser

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define language options and corresponding settings
languages = {
    "English": {
        "stopwords": list(stopwords.words('english')),
        "cleaning_pattern": r'[^a-zA-Z\s]'
    },
    "Spanish": {
        "stopwords": list(stopwords.words('spanish')),
        "cleaning_pattern": r'[^a-zA-Z\sñáéíóúüÁÉÍÓÚÜÑ]'
    },
    "French": {
        "stopwords": list(stopwords.words('french')),
        "cleaning_pattern": r'[^a-zA-Z\sàâçéèêëîïôûùüÿñæœ]'
    },
    "German": {
        "stopwords": list(stopwords.words('german')),
        "cleaning_pattern": r'[^a-zA-Z\säöüßÄÖÜ]'
    },
    "Italian": {
        "stopwords": list(stopwords.words('italian')),
        "cleaning_pattern": r'[^a-zA-Z\sàèéìíîòóùú]'
    }
}

# Function to scrape and clean webpage content
def scrape_webpage_content(url, cleaning_pattern):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ')
        # Basic cleaning: remove non-alphabetic characters, include language-specific characters
        text = re.sub(cleaning_pattern, '', text)
        return text.lower()
    except Exception as e:
        st.write(f"Failed to scrape {url}: {e}")
        return ""

# Function to preprocess text using NLTK
def preprocess_text(text, stopwords):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords]
    return tokens

# Function to compute TF-IDF scores
def compute_tfidf(corpus, stopwords):
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    X = vectorizer.fit_transform(corpus)
    tfidf_scores = X.toarray()
    terms = vectorizer.get_feature_names_out()
    return terms, tfidf_scores

# Convert TF-IDF score to recommended frequency based on the average word count
def convert_tfidf_to_frequency(tfidf_score, total_word_count, avg_word_count, scaling_factor=100):
    frequency = int(tfidf_score * avg_word_count / total_word_count * scaling_factor)
    return max(frequency, 1)  # Ensure at least one occurrence

# Streamlit app
def main():
    st.title("Competitor TF-IDF Analysis")
    
    # Add custom CSS for the selectbox color
    st.markdown(
        """
        <style>
        .stSelectbox label {
            color: #000000 !important;  /* Change text color */
        }
        .stSelectbox div[data-baseweb="select"] {
            background-color: #f0f0f0;  /* Change background color */
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    # Language selection
    selected_language = st.selectbox("Select language", options=list(languages.keys()))
    language_settings = languages[selected_language]
        
    competitors = st.text_area("Enter competitor URLs, separated by commas:")
    
    if st.button("Analyze"):
        
        if not competitors:
            st.error("Please enter at least one competitor URL.")
            return
        
        # Step 1: Get the list of competitor URLs
        urls = [url.strip() for url in competitors.split(',') if url.strip()]

        corpus = []
        word_counts = []
        valid_urls = []  # Track URLs that have valid content

        # Step 2: Scrape content from each result and preprocess it
        for url in urls:
            text = scrape_webpage_content(url, language_settings["cleaning_pattern"])
            if text:  # Only process if there is content
                corpus.append(text)
                word_counts.append(len(text.split()))  # Store the word count of each page
                valid_urls.append(url)  # Only add to valid URLs if content exists

        # Step 3: Calculate the average word count of the top 3 results
        if len(word_counts) >= 3:
            avg_word_count = sum(word_counts[:3]) // 3
        else:
            avg_word_count = sum(word_counts) // len(word_counts)  # Fallback if fewer than 3 results

        # Step 4: Calculate TF-IDF
        if corpus:  # Ensure there is content to process
            terms, tfidf_scores = compute_tfidf(corpus, language_settings["stopwords"])
        else:
            st.write("No valid content to process.")
            return

        total_word_count = sum(len(text.split()) for text in corpus)

        # Step 5: Collect actionable recommendations
        recommendations = []
        for i, url in enumerate(valid_urls):  # Use valid URLs instead of all URLs
            num_terms = len(tfidf_scores[i])  # Get the actual number of terms available
            sorted_indices = tfidf_scores[i].argsort()[::-1][:num_terms]

            for index in sorted_indices:
                term = terms[index]
                score = tfidf_scores[i][index]
                frequency = convert_tfidf_to_frequency(score, total_word_count, avg_word_count)
                recommendations.append({
                    'Keyword': term,
                    'Average TF-IDF Score': score,
                    'Recommended Frequency': frequency,
                    'Source URL': url
                })

        # Step 6: Create the DataFrame and display it
        df = pd.DataFrame(recommendations)
        df.sort_values(by='Average TF-IDF Score', ascending=False, inplace=True)
        # Add a column for 'Average Word Count', and ensure all cells are valid
        df.insert(df.columns.get_loc('Source URL') + 1, 'Average Word Count', avg_word_count)
        df.reset_index(drop=True, inplace=True)

        st.write("### Recommendations")
        st.dataframe(df)

        # Step 7: Convert DataFrame to CSV and provide a download button
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download SEO Recommendations as CSV",
            data=csv,
            file_name="seo_recommendations.csv",
            mime="text/csv",
        )


    # Adding a footnote with a hyperlink to LinkedIn
    st.markdown(
        """
        <div style='text-align: center; padding-top: 20px;'>
            Developed by <a href="https://www.linkedin.com/in/antonio-atilio-maculus-70b6b9196/" target="_blank">Antonio Maculus</a>
        </div>
        """, 
        unsafe_allow_html=True)

if __name__ == "__main__":
    main()