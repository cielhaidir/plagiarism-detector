import pandas as pd
import re
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from tqdm import tqdm

# Initialize tqdm to work with pandas apply
tqdm.pandas()

def preprocess_text(text, stemmer, stopword_remover):
    """
    Cleans and preprocesses a single string of Indonesian text.
    """
    if not isinstance(text, str):
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove URLs, numbers, and non-alphanumeric characters (except spaces)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # 3. Remove extra whitespace
    text = text.strip()
    
    # 4. Stopword removal
    text = stopword_remover.remove(text)
    
    # 5. Stemming
    text = stemmer.stem(text)
    
    return text

def create_indices(file_path):
    """
    Reads the proposal CSV, preprocesses each text column individually,
    and saves a separate index and TF-IDF model for each.
    """
    print("Initializing Indonesian language tools (this may take a moment)...")
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    print("Tools initialized.")

    # Create a directory to store index files
    output_dir = 'indices'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Define columns to be indexed
    text_columns = ['judul', 'ringkasan', 'pendahuluan', 'masalah', 'metode', 'solusi']
    
    # We'll save the non-text data once.
    metadata_df = df[['id', 'skema']].copy()
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Saved metadata to {metadata_path}")

    for column in text_columns:
        print(f"\n--- Processing column: {column} ---")
        
        # 1. Preprocess the text data with a progress bar
        print(f"Applying text preprocessing to '{column}'...")
        processed_texts = df[column].fillna('').progress_apply(lambda x: preprocess_text(x, stemmer, stopword_remover))
        
        # 2. Vectorize the processed text
        print(f"Vectorizing '{column}' with TF-IDF...")
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        # 3. Save the vectorizer and the matrix
        vectorizer_path = os.path.join(output_dir, f'vectorizer_{column}.pkl')
        matrix_path = os.path.join(output_dir, f'tfidf_matrix_{column}.pkl')
        
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(tfidf_matrix, matrix_path)
        
        print(f"Successfully saved index for '{column}'")
        print(f"  - Vectorizer: {vectorizer_path}")
        print(f"  - Matrix: {matrix_path}")

    print("\n--- All columns have been processed and indexed. ---")

if __name__ == '__main__':
    create_indices('proposal.csv')