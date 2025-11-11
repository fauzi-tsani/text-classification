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
    try:
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        cleansed_text = cleanse_text(text)
        if not cleansed_text:
            return ""
            
        tokens = tokenize_text(cleansed_text)
        if not tokens:
            return ""
            
        filtered_tokens = filter_stopwords(tokens)
        if not filtered_tokens:
            return ""
            
        # Skip stemming if too many tokens to avoid performance issues
        if len(filtered_tokens) > 1000:
            return ' '.join(filtered_tokens[:1000])
            
        stemmed_tokens = stem_text(filtered_tokens)
        return ' '.join(stemmed_tokens)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return text if isinstance(text, str) else ""

if __name__ == "__main__":
    print("Starting text preprocessing...")
    
    # Load the dataset
    df = pd.read_csv('news_articles.csv')
    
    print(f"Loaded {len(df)} articles for preprocessing")
    
    # Apply preprocessing to the 'content' column with progress tracking
    processed_count = 0
    cleaned_contents = []
    
    for idx, content in enumerate(df['content']):
        try:
            cleaned_content = preprocess_text(content)
            cleaned_contents.append(cleaned_content)
            processed_count += 1
            
            # Print progress every 10 articles
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(df)} articles...")
                
        except Exception as e:
            print(f"Error processing article {idx}: {e}")
            cleaned_contents.append("")
    
    # Add the cleaned content to the dataframe
    df['cleaned_content'] = cleaned_contents
    
    # Remove rows with empty cleaned content
    df = df[df['cleaned_content'].str.len() > 0]
    
    print(f"Successfully processed {len(df)} articles")
    
    # Save the processed data to a new CSV file
    df.to_csv('cleaned_news_articles.csv', index=False)
    
    print("Text preprocessing completed. Cleaned data saved to 'cleaned_news_articles.csv'")
    print(f"Sample cleaned content: {df['cleaned_content'].iloc[0][:200]}...")