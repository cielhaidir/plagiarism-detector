import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio as levenshtein_ratio
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a multilingual model suitable for Indonesian
_sentence_model = None

def initialize_sentence_model():
    """Pre-load the sentence transformer model at startup."""
    global _sentence_model
    if _sentence_model is None:
        print("Loading sentence transformer model (this may take a moment)...")
        _sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        print("Sentence transformer model loaded successfully!")
    return _sentence_model

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        # Fallback - should not happen if initialized properly
        _sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    return _sentence_model

def ngrams(text, n=3):
    """Extract word n-grams from text."""
    tokens = re.findall(r'\w+', text.lower())
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)) if len(tokens) >= n else set()

def jaccard_similarity(text1, text2, n=3):
    """Jaccard similarity over word n-grams."""
    ngrams1 = ngrams(text1, n)
    ngrams2 = ngrams(text2, n)
    if not ngrams1 or not ngrams2:
        return 0.0
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    return len(intersection) / len(union)

def levenshtein_similarity(text1, text2):
    """Levenshtein similarity ratio (0-1)."""
    return levenshtein_ratio(text1, text2)

def tfidf_cosine_similarity(text1, text2, vectorizer=None):
    """TF-IDF cosine similarity between two texts."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([text1, text2])
    else:
        X = vectorizer.transform([text1, text2])
    return cosine_similarity(X[0], X[1])[0][0]

def sentence_embedding_similarity(text1, text2):
    """Cosine similarity between sentence embeddings."""
    model = get_sentence_model()
    emb = model.encode([text1, text2])
    sim = np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))
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
        weights = (0.2, 0.4, 0.4)  # Boost fuzzy and semantic
    # Medium texts have a balanced approach
    elif 100 <= len1 < 500 or 100 <= len2 < 500:
        weights = (0.3, 0.3, 0.4)  # Standard balanced weights
    # Longer texts can rely more on exact (n-gram) similarity
    else:
        weights = (0.5, 0.2, 0.3)  # Boost exact match for long texts

    # Calculate the weighted score
    final_score = (weights[0] * exact) + (weights[1] * fuzzy) + (weights[2] * semantic)
    
    # Add a bonus for high semantic similarity, as it's a strong indicator
    if semantic > 0.9:
        final_score += 0.05
        
    # Add a small bonus for high fuzzy similarity
    if fuzzy > 0.9:
        final_score += 0.03

    # Ensure the score does not exceed 1.0
    return min(final_score, 1.0)