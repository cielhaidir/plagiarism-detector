import pandas as pd
import re
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from tqdm import tqdm
import numpy as np

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
        
        # 2. Save the preprocessed texts to CSV for faster search operations
        processed_output_path = os.path.join(output_dir, f'processed_{column}.csv')
        processed_texts.to_csv(processed_output_path, index=False, header=False)
        print(f"Saved preprocessed texts to {processed_output_path}")
        
        # 3. Vectorize the processed text
        print(f"Vectorizing '{column}' with TF-IDF...")
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        # 4. Save the vectorizer and the matrix
        vectorizer_path = os.path.join(output_dir, f'vectorizer_{column}.pkl')
        matrix_path = os.path.join(output_dir, f'tfidf_matrix_{column}.pkl')
        
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(tfidf_matrix, matrix_path)
        
        print(f"Successfully saved index for '{column}'")
        print(f"  - Vectorizer: {vectorizer_path}")
        print(f"  - Matrix: {matrix_path}")
        print(f"  - Preprocessed texts: {processed_output_path}")

    print("\n--- All columns have been processed and indexed. ---")

def append_indices(new_proposals_df, year_threshold=2025):
    """
    Append new proposals to existing indices with year filtering.
    
    Args:
        new_proposals_df: DataFrame containing new proposals
        year_threshold: Minimum year to include (default: 2025)
    
    Returns:
        dict: Processing results with counts and errors
    """
    print("Initializing Indonesian language tools...")
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    print("Tools initialized.")

    output_dir = 'indices'
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure tahun column exists for filtering
    if 'tahun' not in new_proposals_df.columns:
        raise ValueError("DataFrame must contain 'tahun' column for year filtering")
    
    # Filter by year threshold
    filtered_df = new_proposals_df[new_proposals_df['tahun'] >= year_threshold].copy()
    filtered_count = len(filtered_df)
    total_count = len(new_proposals_df)
    
    if filtered_count == 0:
        return {
            'status': 'no_data',
            'message': f'No proposals found with tahun >= {year_threshold}',
            'total_proposals': total_count,
            'filtered_proposals': 0,
            'processed_proposals': 0
        }
    
    print(f"Processing {filtered_count} proposals (filtered from {total_count})")
    
    # Define columns to be indexed
    text_columns = ['judul', 'ringkasan', 'pendahuluan', 'masalah', 'metode', 'solusi']
    
    # Load existing metadata
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    if os.path.exists(metadata_path):
        existing_metadata = pd.read_csv(metadata_path)
        # Check for duplicate IDs
        duplicate_ids = set(filtered_df['id']).intersection(set(existing_metadata['id']))
        if duplicate_ids:
            filtered_df = filtered_df[~filtered_df['id'].isin(duplicate_ids)]
            print(f"Skipped {len(duplicate_ids)} duplicate IDs")
    
    if len(filtered_df) == 0:
        return {
            'status': 'no_new_data',
            'message': 'All proposals were duplicates or filtered out',
            'total_proposals': total_count,
            'filtered_proposals': filtered_count,
            'processed_proposals': 0,
            'duplicate_count': len(duplicate_ids) if 'duplicate_ids' in locals() else 0
        }
    
    # Append new metadata
    new_metadata = filtered_df[['id', 'skema']].copy()
    if os.path.exists(metadata_path):
        combined_metadata = pd.concat([existing_metadata, new_metadata], ignore_index=True)
    else:
        combined_metadata = new_metadata
    
    combined_metadata.to_csv(metadata_path, index=False)
    print(f"Updated metadata with {len(new_metadata)} new entries")
    
    # Process each text column
    processed_counts = {}
    for column in text_columns:
        print(f"\n--- Processing column: {column} ---")
        
        # Load existing data if exists
        processed_output_path = os.path.join(output_dir, f'processed_{column}.csv')
        vectorizer_path = os.path.join(output_dir, f'vectorizer_{column}.pkl')
        matrix_path = os.path.join(output_dir, f'tfidf_matrix_{column}.pkl')
        
        # Preprocess new texts
        new_processed = filtered_df[column].fillna('').progress_apply(
            lambda x: preprocess_text(x, stemmer, stopword_remover)
        )
        
        # Load existing processed texts if they exist
        if os.path.exists(processed_output_path):
            existing_processed = pd.read_csv(processed_output_path, header=None)[0]
            combined_processed = pd.concat([existing_processed, new_processed], ignore_index=True)
        else:
            combined_processed = new_processed
        
        # Save updated processed texts
        combined_processed.to_csv(processed_output_path, index=False, header=False)
        
        # Update TF-IDF vectorizer and matrix
        print(f"Updating TF-IDF for '{column}'...")
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(combined_processed)
        
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(tfidf_matrix, matrix_path)
        
        processed_counts[column] = len(new_processed)
        print(f"Updated {column} with {len(new_processed)} new entries")
    
    return {
        'status': 'success',
        'message': 'Indices updated successfully',
        'total_proposals': total_count,
        'filtered_proposals': filtered_count,
        'processed_proposals': len(filtered_df),
        'duplicate_count': len(duplicate_ids) if 'duplicate_ids' in locals() else 0,
        'columns_updated': processed_counts
    }

if __name__ == '__main__':
    create_indices('skripsi_with_skema.csv')
