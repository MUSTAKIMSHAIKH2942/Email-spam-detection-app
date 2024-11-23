import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

def preprocess_text(text):
    """
    Cleans and preprocesses email text.
    Steps:
    - Remove special characters and digits
    - Convert text to lowercase
    - Remove stopwords
    - Apply stemming
    """
    # Initialize stemmer and stopwords
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Remove special characters and digits
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)

    # Convert to lowercase and tokenize
    tokens = text.lower().split()

    # Remove stopwords and apply stemming
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]

    return ' '.join(tokens)



def load_and_preprocess_data(filepath):
    """
    Loads dataset, preprocesses email text, and prepares features/labels.
    """
    # Load dataset
    df = pd.read_csv(filepath, encoding='latin-1')

    # Use columns v1 (label) and v2 (text)
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

    # Encode labels: spam = 1, ham = 0
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    # Preprocess email text
    df['text'] = df['text'].apply(preprocess_text)

    return df['text'], df['label']

