from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import os
from similarity_utils import (
    jaccard_similarity,
    levenshtein_similarity,
    tfidf_cosine_similarity,
    sentence_embedding_similarity,
    calculate_final_score,
    initialize_sentence_model
)

app = Flask(__name__)

# Global variables to store loaded indices
indices = {}
metadata = None
stemmer = None
stopword_remover = None
# Load original proposal texts for matched_text retrieval
original_texts = None

def preprocess_text(text, stemmer, stopword_remover):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

def load_indices():
    global indices, metadata, stemmer, stopword_remover, original_texts
    print("Loading Indonesian language tools...")
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    print("Loading metadata...")
    metadata = pd.read_csv('indices/metadata.csv')
    text_columns = ['judul', 'ringkasan', 'pendahuluan', 'masalah', 'metode', 'solusi']
    print("Loading indices for each column...")
    for column in text_columns:
        print(f"  Loading {column}...")
        vectorizer_path = f'indices/vectorizer_{column}.pkl'
        matrix_path = f'indices/tfidf_matrix_{column}.pkl'
        vectorizer = joblib.load(vectorizer_path)
        tfidf_matrix = joblib.load(matrix_path)
        indices[column] = {
            'vectorizer': vectorizer,
            'matrix': tfidf_matrix
        }
    print("Loading original proposal texts for matched_text retrieval...")
    original_texts = pd.read_csv('proposal.csv')
    print("Pre-loading sentence transformer model for faster queries...")
    initialize_sentence_model()
    print("All indices and models loaded successfully!")

def search_column(query_text, column, skema_filter=None, top_k=10):
    global indices, metadata, stemmer, stopword_remover, original_texts
    if column not in indices:
        return []
    processed_query = preprocess_text(query_text, stemmer, stopword_remover)
    vectorizer = indices[column]['vectorizer']
    query_vector = vectorizer.transform([processed_query])
    similarity_matrix = cosine_similarity(query_vector, indices[column]['matrix'])
    similarities = similarity_matrix[0]
    sorted_indices = np.argsort(similarities)[::-1]
    results = []
    for idx in sorted_indices:
        similarity_score = similarities[idx]
        if similarity_score < 0.1:
            break
        proposal_id = metadata.iloc[idx]['id']
        proposal_skema = metadata.iloc[idx]['skema']
        if skema_filter and proposal_skema != skema_filter:
            continue

        # Retrieve the matched text for this proposal and column
        matched_row = original_texts[original_texts['id'] == proposal_id]
        matched_text = ""
        if not matched_row.empty and column in matched_row.columns:
            matched_text = str(matched_row.iloc[0][column]) if pd.notna(matched_row.iloc[0][column]) else ""

        # Calculate advanced similarity metrics
        exact_score = jaccard_similarity(query_text, matched_text)
        fuzzy_score = levenshtein_similarity(query_text, matched_text)
        semantic_score = sentence_embedding_similarity(query_text, matched_text)
        final_score = calculate_final_score(exact_score, fuzzy_score, semantic_score)

        results.append({
            'id': int(proposal_id),
            'skema': proposal_skema,
            'similarity_score': float(similarity_score),  # legacy TF-IDF score
            'exact_score': float(exact_score),
            'fuzzy_score': float(fuzzy_score),
            'semantic_score': float(semantic_score),
            'final_score': float(final_score),
            'column': column,
            'matched_text': matched_text[:500] + "..." if len(matched_text) > 500 else matched_text  # Truncate for API response
        })
        if len(results) >= top_k:
            break
    return results

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        query_text = data.get('query_text')
        column = data.get('column')
        skema_filter = data.get('skema')
        top_k = data.get('top_k', 10)
        if not query_text:
            return jsonify({"error": "query_text is required"}), 400
        if not column:
            return jsonify({"error": "column is required"}), 400
        if column not in indices:
            valid_columns = list(indices.keys())
            return jsonify({"error": f"Invalid column. Valid columns are: {valid_columns}"}), 400
        results = search_column(query_text, column, skema_filter, top_k)
        return jsonify({
            "results": results,
            "query_info": {
                "column": column,
                "skema_filter": skema_filter,
                "total_results": len(results)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search_bulk', methods=['POST'])
def search_bulk():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        texts = data.get('texts', [])
        top_k = data.get('top_k', 5)
        if not texts:
            return jsonify({"error": "texts array is required"}), 400
        bulk_results = []
        for i, text_item in enumerate(texts):
            print(f"[BULK] Processing query {i+1}/{len(texts)}")
            # Handle case where text_item is a list containing a dict
            if isinstance(text_item, list):
                if len(text_item) > 0 and isinstance(text_item[0], dict):
                    text_item = text_item[0]  # Extract the dict from the list
                else:
                    bulk_results.append({
                        "query_index": i,
                        "error": f"Invalid list format: {text_item}"
                    })
                    continue
            
            # Check if it's a dictionary
            if not isinstance(text_item, dict):
                bulk_results.append({
                    "query_index": i,
                    "error": f"Expected dict but got {type(text_item).__name__}: {text_item}"
                })
                continue
                
            skema_filter = text_item.get('skema')
            item_results = []
            # Loop through all possible columns
            for column in indices.keys():
                query_text = text_item.get(column)
                if query_text:
                    print(f"[BULK]   - Searching column '{column}' for query {i+1}")
                    results = search_column(query_text, column, skema_filter, top_k)
                    for result in results:
                        result['searched_column'] = column
                    item_results.extend(results)
            # Sort and limit results
            item_results.sort(key=lambda x: x['final_score'], reverse=True)
            item_results = item_results[:top_k]
            print(f"[BULK]   - Finished query {i+1}, found {len(item_results)} results")
            bulk_results.append({
                "query_index": i,
                "results": item_results,
                "query_info": {
                    "skema_filter": skema_filter,
                    "columns_searched": [col for col in indices.keys() if text_item.get(col)],
                    "total_results": len(item_results)
                }
            })
        return jsonify({
            "bulk_results": bulk_results,
            "total_queries": len(texts)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "indices_loaded": len(indices) > 0})

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "available_columns": list(indices.keys()),
        "total_proposals": len(metadata) if metadata is not None else 0,
        "unique_skemas": list(metadata['skema'].unique()) if metadata is not None else []
    })

if __name__ == '__main__':
    print("Starting plagiarism detection API...")
    load_indices()
    print("API server ready!")
    app.run(debug=True, host='0.0.0.0', port=5000)