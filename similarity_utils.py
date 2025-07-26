import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio as levenshtein_ratio
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
import pickle
import os
from functools import lru_cache

# Load a multilingual model suitable for Indonesian
_sentence_model = None
_embedding_cache = {}
_cache_file = "embeddings_cache.pkl"

def load_embedding_cache():
    """Load embedding cache from disk."""
    global _embedding_cache
    if os.path.exists(_cache_file):
        try:
            with open(_cache_file, 'rb') as f:
                _embedding_cache = pickle.load(f)
            print(f"Loaded {len(_embedding_cache)} cached embeddings")
        except Exception as e:
            print(f"Failed to load embedding cache: {e}")
            _embedding_cache = {}
    else:
        _embedding_cache = {}

def save_embedding_cache():
    """Save embedding cache to disk."""
    try:
        with open(_cache_file, 'wb') as f:
            pickle.dump(_embedding_cache, f)
        print(f"Saved {len(_embedding_cache)} embeddings to cache")
    except Exception as e:
        print(f"Failed to save embedding cache: {e}")

def cleanup_caches():
    """Save caches and cleanup on shutdown."""
    save_embedding_cache()
    # Clear LRU caches to free memory
    ngrams_cached.cache_clear()
    jaccard_similarity_cached.cache_clear()
    levenshtein_similarity_cached.cache_clear()

def get_text_hash(text):
    """Get a hash for text to use as cache key."""
    if not text or not isinstance(text, str):
        return ""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def initialize_sentence_model():
    """Pre-load the sentence transformer model at startup."""
    global _sentence_model
    if _sentence_model is None:
        print("Loading sentence transformer model (this may take a moment)...")
        _sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        print("Sentence transformer model loaded successfully!")
        # Load cached embeddings
        load_embedding_cache()
    return _sentence_model

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        # Fallback - should not happen if initialized properly
        _sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    return _sentence_model

@lru_cache(maxsize=5000)
def ngrams_cached(text, n=3):
    """Extract word n-grams from text with caching."""
    if not text or not isinstance(text, str):
        return frozenset()
    tokens = re.findall(r'\w+', text.lower())
    return frozenset(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)) if len(tokens) >= n else frozenset()

def ngrams(text, n=3):
    """Extract word n-grams from text."""
    return ngrams_cached(text, n)

@lru_cache(maxsize=5000)
def jaccard_similarity_cached(text1, text2, n=3):
    """Jaccard similarity over word n-grams with caching."""
    if not text1 or not text2:
        return 0.0
    ngrams1 = ngrams_cached(text1, n)
    ngrams2 = ngrams_cached(text2, n)
    if not ngrams1 or not ngrams2:
        return 0.0
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    return len(intersection) / len(union)

def jaccard_similarity(text1, text2, n=3):
    """Jaccard similarity over word n-grams."""
    return jaccard_similarity_cached(text1, text2, n)

@lru_cache(maxsize=5000)
def levenshtein_similarity_cached(text1, text2):
    """Levenshtein similarity ratio (0-1) with caching."""
    if not text1 or not text2:
        return 0.0
    return levenshtein_ratio(text1, text2)

def levenshtein_similarity(text1, text2):
    """Levenshtein similarity ratio (0-1)."""
    return levenshtein_similarity_cached(text1, text2)

def tfidf_cosine_similarity(text1, text2, vectorizer=None):
    """TF-IDF cosine similarity between two texts."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([text1, text2])
    else:
        X = vectorizer.transform([text1, text2])
    return cosine_similarity(X[0], X[1])[0][0]

def get_cached_embedding(text):
    """Get cached embedding or compute and cache it."""
    global _embedding_cache, _sentence_model
    
    if not text or not isinstance(text, str):
        return np.zeros(512)  # Default embedding size
    
    text_hash = get_text_hash(text)
    
    # Check cache first
    if text_hash in _embedding_cache:
        return _embedding_cache[text_hash]
    
    # Compute embedding if not cached
    if _sentence_model is None:
        model = get_sentence_model()
    else:
        model = _sentence_model
    
    embedding = model.encode([text])[0]
    
    # Cache the result (limit cache size to prevent memory issues)
    if len(_embedding_cache) > 10000:  # Limit cache size
        # Remove 20% of oldest entries
        oldest_keys = list(_embedding_cache.keys())[:2000]
        for key in oldest_keys:
            del _embedding_cache[key]
    
    _embedding_cache[text_hash] = embedding
    
    # Periodically save cache to disk
    if len(_embedding_cache) % 100 == 0:
        save_embedding_cache()
    
    return embedding

def sentence_embedding_similarity(text1, text2):
    """Cosine similarity between sentence embeddings with caching."""
    emb1 = get_cached_embedding(text1)
    emb2 = get_cached_embedding(text2)
    
    # Handle edge cases
    if np.allclose(emb1, 0) or np.allclose(emb2, 0):
        return 0.0
    
    # Compute cosine similarity
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(sim)

def calculate_final_score(exact, fuzzy, semantic, text1="", text2=""):
    """
    Calculates a final similarity score using an adaptive weighted algorithm.
    The weights for exact, fuzzy, and semantic scores are adjusted based on
    the length of the texts being compared.
    """
    len1 = len(text1)
    len2 = len(text2)

    # Shorter texts rely more on semantic and fuzzy similarity
    if len1 < 100 or len2 < 100:
        weights = (0.4, 0.0, 0.6)  # Boost fuzzy and semantic
    # Medium texts have a balanced approach
    elif 100 <= len1 < 500 or 100 <= len2 < 500:
        weights = (0.5, 0.0, 0.5)  # Standard balanced weights
    # Longer texts can rely more on exact (n-gram) similarity
    else:
        weights = (0.5, 0.0, 0.5)  # Boost exact match for long texts

    # Calculate the weighted score
    final_score = (weights[0] * exact) + (weights[1] * fuzzy) + (weights[2] * semantic)
    
    # Add a bonus for high semantic similarity, as it's a strong indicator
    if semantic > 0.9:
        final_score += 0.05
        
    # Add a small bonus for high fuzzy similarity
    if fuzzy > 0.9:
        final_score += 0.00

    # Ensure the score does not exceed 1.0
    return min(final_score, 1.0)