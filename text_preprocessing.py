import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def cleanse_text(text):
    """Remove special characters, numbers, and extra whitespace from the text."""
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove leading/trailing whitespace
    return text

def tokenize_text(text):
    """Tokenize the text into individual words."""
    return word_tokenize(text)

def filter_stopwords(tokens):
    """Remove stopwords from the list of tokens."""
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in tokens if word not in stop_words]

def stem_text(tokens):
    """Perform stemming on the list of tokens."""
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(word) for word in tokens]

def preprocess_text(text):
    """Perform the complete text preprocessing pipeline."""
    cleansed_text = cleanse_text(text)
    tokens = tokenize_text(cleansed_text)
    filtered_tokens = filter_stopwords(tokens)
    stemmed_tokens = stem_text(filtered_tokens)
    return ' '.join(stemmed_tokens)

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('news_articles.csv')

    # Apply preprocessing to the 'content' column
    df['cleaned_content'] = df['content'].apply(preprocess_text)

    # Save the processed data to a new CSV file
    df.to_csv('cleaned_news_articles.csv', index=False)

    print("Text preprocessing completed. Cleaned data saved to 'cleaned_news_articles.csv'")