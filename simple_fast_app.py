from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import hashlib

app = Flask(__name__)

# Global variables
model = None
precomputed_embeddings = {}
metadata = None
original_texts = None

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'  # Small, fast model
EMBEDDINGS_DIR = 'precomputed_embeddings'

def load_model():
    global model
    print("Loading fast sentence transformer...")
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    print("✓ Model loaded")

@lru_cache(maxsize=1000)
def get_text_hash(text):
    """Get hash for text caching"""
    return hashlib.md5(str(text).encode()).hexdigest()

@lru_cache(maxsize=2000)
def get_embedding_cached(text):
    """Get cached embedding for text"""
    global model
    if not text or len(str(text).strip()) == 0:
        return None
    
    # Use the model to encode
    embedding = model.encode([str(text)], convert_to_tensor=False)[0]
    return embedding.astype(np.float32)

def load_precomputed_embeddings():
    """Load precomputed embeddings if they exist"""
    global precomputed_embeddings
    
    text_columns = ['judul', 'ringkasan', 'pendahuluan', 'masalah', 'metode', 'solusi']
    
    for column in text_columns:
        embedding_file = f"{EMBEDDINGS_DIR}/{column}_embeddings.pkl"
        if os.path.exists(embedding_file):
            try:
                with open(embedding_file, 'rb') as f:
                    precomputed_embeddings[column] = pickle.load(f)
                print(f"✓ Loaded precomputed embeddings for {column}")
            except Exception as e:
                print(f"Error loading {column} embeddings: {e}")
        else:
            print(f"No precomputed embeddings for {column}, will compute on demand")

def load_data():
    global metadata, original_texts
    print("Loading data...")
    try:
        metadata = pd.read_csv('indices/metadata.csv')
        original_texts = pd.read_csv('skripsi_with_skema.csv')
        print("✓ Data loaded")
    except Exception as e:
        print(f"Error loading data: {e}")
        metadata = pd.DataFrame()
        original_texts = pd.DataFrame()

def fast_similarity_search(query_text, column, top_k=10, skema_filter=None):
    """Fast similarity search using precomputed or cached embeddings"""
    global precomputed_embeddings, original_texts
    
    if column not in original_texts.columns:
        return []
    
    # Get query embedding
    query_embedding = get_embedding_cached(query_text)
    if query_embedding is None:
        return []
    
    query_embedding = query_embedding.reshape(1, -1)
    
    results = []
    
    # Check if we have precomputed embeddings for this column
    if column in precomputed_embeddings:
        # Use precomputed embeddings (fastest)
        embeddings_data = precomputed_embeddings[column]
        
        for idx, item in embeddings_data.items():
            if skema_filter and item.get('skema') != skema_filter:
                continue
                
            doc_embedding = item['embedding'].reshape(1, -1)
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            
            if similarity > 0.1:  # Filter very low similarities
                results.append({
                    'id': item.get('id'),
                    'skema': item.get('skema'),
                    'similarity_score': float(similarity),
                    'column': column,
                    'matched_text': item.get('text', '')[:500]
                })
    else:
        # Compute embeddings on demand (slower but still works)
        for idx, row in original_texts.iterrows():
            text = row.get(column)
            if pd.notna(text) and isinstance(text, str) and len(text.strip()) > 0:
                
                if skema_filter and row.get('skema') != skema_filter:
                    continue
                
                # Get embedding for this text
                doc_embedding = get_embedding_cached(text)
                if doc_embedding is not None:
                    doc_embedding = doc_embedding.reshape(1, -1)
                    similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
                    
                    if similarity > 0.1:
                        results.append({
                            'id': row.get('id', idx),
                            'skema': row.get('skema'),
                            'similarity_score': float(similarity),
                            'column': column,
                            'matched_text': str(text)[:500]
                        })
    
    # Sort by similarity and return top_k
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results[:top_k]

@app.route('/search', methods=['POST'])
def search():
    """Fast search endpoint"""
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
        
        start_time = time.time()
        results = fast_similarity_search(query_text, column, top_k, skema_filter)
        search_time = time.time() - start_time
        
        return jsonify({
            "results": results,
            "search_time": round(search_time, 4),
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
    """Fast bulk search endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        texts = data.get('texts', [])
        top_k = data.get('top_k', 1)
        
        if not texts:
            return jsonify({"error": "texts array is required"}), 400
        
        start_time = time.time()
        bulk_results = []
        
        for i, text_item in enumerate(texts):
            print(f"[SIMPLE-FAST] Processing query {i+1}/{len(texts)}")
            
            if isinstance(text_item, list) and len(text_item) > 0:
                text_item = text_item[0]
            
            if not isinstance(text_item, dict):
                bulk_results.append({
                    "query_index": i,
                    "error": "Invalid format"
                })
                continue
            
            skema_filter = text_item.get('skema')
            item_results = []
            
            # Search available columns
            available_columns = ['judul', 'ringkasan', 'pendahuluan', 'masalah', 'metode', 'solusi']
            for column in available_columns:
                query_text = text_item.get(column)
                if query_text:
                    results = fast_similarity_search(query_text, column, top_k, skema_filter)
                    for result in results:
                        result['searched_column'] = column
                    item_results.extend(results)
            
            # Sort by similarity score
            item_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            item_results = item_results[:top_k]
            
            bulk_results.append({
                "query_index": i,
                "results": item_results,
                "query_info": {
                    "skema_filter": skema_filter,
                    "columns_searched": [col for col in available_columns if text_item.get(col)],
                    "total_results": len(item_results)
                }
            })
        
        total_time = time.time() - start_time
        
        return jsonify({
            "bulk_results": bulk_results,
            "total_queries": len(texts),
            "total_time": round(total_time, 4),
            "avg_time_per_query": round(total_time / len(texts), 4)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "precomputed_columns": list(precomputed_embeddings.keys()),
        "cache_info": {
            "embedding_cache_size": get_embedding_cached.cache_info().currsize if hasattr(get_embedding_cached, 'cache_info') else 0,
            "hash_cache_size": get_text_hash.cache_info().currsize if hasattr(get_text_hash, 'cache_info') else 0
        }
    })

@app.route('/info', methods=['GET'])
def info():
    """Info endpoint"""
    global metadata
    return jsonify({
        "available_columns": ['judul', 'ringkasan', 'pendahuluan', 'masalah', 'metode', 'solusi'],
        "precomputed_columns": list(precomputed_embeddings.keys()),
        "total_proposals": len(metadata) if metadata is not None else 0,
        "unique_skemas": list(metadata['skema'].unique()) if metadata is not None and 'skema' in metadata.columns else [],
        "model_name": MODEL_NAME,
        "search_type": "Cached Embeddings + Cosine Similarity"
    })

@app.route('/precompute', methods=['POST'])
def precompute_embeddings():
    """Precompute embeddings for faster searches"""
    try:
        data = request.get_json()
        column = data.get('column') if data else None
        
        if not column:
            return jsonify({"error": "column parameter required"}), 400
        
        if column not in original_texts.columns:
            return jsonify({"error": f"Column {column} not found"}), 400
        
        # Precompute embeddings for this column
        print(f"Precomputing embeddings for {column}...")
        embeddings_data = {}
        
        for idx, row in original_texts.iterrows():
            text = row.get(column)
            if pd.notna(text) and isinstance(text, str) and len(text.strip()) > 0:
                embedding = get_embedding_cached(text)
                if embedding is not None:
                    embeddings_data[idx] = {
                        'id': row.get('id', idx),
                        'text': str(text),
                        'skema': row.get('skema'),
                        'embedding': embedding
                    }
        
        # Save precomputed embeddings
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        embedding_file = f"{EMBEDDINGS_DIR}/{column}_embeddings.pkl"
        
        with open(embedding_file, 'wb') as f:
            pickle.dump(embeddings_data, f)
        
        # Update global cache
        precomputed_embeddings[column] = embeddings_data
        
        return jsonify({
            "message": f"Precomputed {len(embeddings_data)} embeddings for {column}",
            "saved_to": embedding_file
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Initialize on startup
print("Initializing Simple Fast App...")
load_model()
load_data()
load_precomputed_embeddings()
print("✓ Simple Fast App ready!")

if __name__ == '__main__':
    print("Starting SIMPLE Fast Plagiarism Detection API...")
    print("This version avoids FAISS but still provides major speedup through caching")
    print("API server ready!")
    app.run(debug=True, host='0.0.0.0', port=5002)  # Port 5002 to avoid conflicts