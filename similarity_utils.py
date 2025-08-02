import re
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio as levenshtein_ratio
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a multilingual model suitable for Indonesian
_sentence_model = None

def get_optimal_device():
    """Determine the best available device, with careful MPS testing for Intel Macs with AMD GPUs."""
    
    # First check CUDA (unlikely on Mac but check anyway)
    if torch.cuda.is_available():
        print("CUDA GPU detected, using CUDA")
        return 'cuda'
    
    # For Intel Macs with AMD GPUs, try MPS carefully
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            print("Testing MPS (Metal Performance Shaders) compatibility...")
            
            # Test with a small tensor first
            test_tensor = torch.tensor([1.0, 2.0, 3.0], device='mps')
            result = test_tensor * 2
            cpu_result = result.cpu()
            
            # Test sentence transformer compatibility
            print("Testing sentence transformer with MPS...")
            test_model = SentenceTransformer('all-MiniLM-L6-v2', device='mps')
            test_embedding = test_model.encode(["test sentence"], show_progress_bar=False)
            
            print("MPS compatibility test passed! Using AMD Radeon Pro 5500M via MPS")
            return 'mps'
            
        except Exception as e:
            print(f"MPS test failed ({str(e)[:100]}...), falling back to CPU")
            print("This is common on Intel Macs with dual GPUs due to driver limitations")
            return 'cpu'
    else:
        print("MPS not available, using CPU")
        return 'cpu'

def initialize_sentence_model(force_cpu=False):
    """Pre-load the sentence transformer model at startup with proper device handling."""
    global _sentence_model
    if _sentence_model is None:
        print("Loading sentence transformer model (this may take a moment)...")
        
        # Get optimal device for this system, but allow forcing CPU for worker processes
        device = 'cpu' if force_cpu else get_optimal_device()
        
        try:
            # Initialize with explicit device and disable progress bars
            _sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2', device=device)
            # Disable progress bars for encoding
            _sentence_model.encode_kwargs = {'show_progress_bar': False}
            print(f"Sentence transformer model loaded successfully on {device}!")
        except Exception as e:
            print(f"Failed to load model on {device}, falling back to CPU: {e}")
            try:
                _sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2', device='cpu')
                _sentence_model.encode_kwargs = {'show_progress_bar': False}
                print("Sentence transformer model loaded successfully on CPU!")
            except Exception as cpu_error:
                print(f"Failed to load model even on CPU: {cpu_error}")
                raise cpu_error
    return _sentence_model

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        # Fallback - should not happen if initialized properly
        print("Sentence model not initialized, initializing now...")
        return initialize_sentence_model()
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
    emb = model.encode([text1, text2], show_progress_bar=False)
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