# Dokumentasi Metode Pencarian Similaritas

## Ringkasan Proyek

Proyek ini mengimplementasikan dua metode pencarian similaritas untuk deteksi plagiarisme proposal penelitian Indonesia. Kedua metode memiliki pendekatan yang berbeda namun saling melengkapi untuk memberikan hasil yang akurat.

---

## Metode 1: Qdrant Vector Search

### Deskripsi
Metode Qdrant menggunakan vector database dan sentence embeddings untuk melakukan pencarian similaritas semantic yang cepat dan akurat. Metode ini memanfaatkan teknologi AI modern dengan model transformers multilingual.

### File Terkait
- [`fast_api.py`](fast_api.py:1) - API endpoint utama
- [`qdrant_search.py`](qdrant_search.py:1) - Implementasi logika pencarian Qdrant

### Arsitektur Teknis

#### 1. Model dan Preprocessing
```python
# Model sentence transformer multilingual
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Preprocessing teks Indonesia
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Hapus URL
    text = re.sub(r'\d+', '', text)                       # Hapus angka
    text = re.sub(r'[^\w\s]', '', text)                   # Hapus tanda baca
    return text.strip()
```

#### 2. Vector Database Setup
```python
# Konfigurasi Qdrant collection
self.client.recreate_collection(
    collection_name="indonesian_proposals",
    vectors_config=VectorParams(
        size=512,  # Dimensi embedding dari model
        distance=Distance.COSINE  # Metrik jarak cosine
    )
)
```

#### 3. Indexing Process
- **Input**: File CSV [`skripsi_with_skema.csv`](skripsi_with_skema.csv:1) 
- **Kolom yang diindeks**: `['judul', 'ringkasan', 'pendahuluan', 'masalah', 'metode', 'solusi']`
- **Proses**:
  1. Preprocessing teks per kolom
  2. Generate embedding menggunakan [`SentenceTransformer`](qdrant_search.py:33)
  3. Store embedding ke Qdrant dengan metadata (proposal_id, skema, column, text)

#### 4. Search Process
```python
def search(self, query_text: str, column: str = None, 
           skema_filter: str = None, limit: int = 10, 
           threshold: float = 0.7) -> List[Dict[str, Any]]:
    # 1. Generate query embedding
    query_embedding = self.create_embeddings([query_text])[0]
    
    # 2. Vector similarity search dengan filter
    search_result = self.client.search(
        collection_name=self.collection_name,
        query_vector=query_embedding,
        query_filter=search_filter,
        limit=limit,
        score_threshold=threshold
    )
    
    # 3. Format dan highlight hasil
    return formatted_results
```

#### 5. Text Highlighting Feature
Menggunakan [`difflib.SequenceMatcher`](qdrant_search.py:77) untuk menandai bagian teks yang mirip:
```python
def _highlight_similarities(self, query_text: str, matched_text: str) -> str:
    # Menggunakan algoritma sequence matching untuk highlight
    matcher = difflib.SequenceMatcher(None, query_words, matched_words)
    # Menandai kata/frasa yang cocok dengan [bracket]
    return highlighted_text
```

### Keunggulan Metode Qdrant
1. **Pencarian Semantic**: Memahami makna kontekstual, bukan hanya kecocokan kata
2. **Kecepatan Tinggi**: Vector search sangat cepat untuk dataset besar
3. **Skalabilitas**: Qdrant dapat handle jutaan dokumen
4. **Akurasi Tinggi**: Model multilingual terlatih untuk bahasa Indonesia
5. **Real-time Search**: Hasil instan untuk query apapun

### Endpoint API
- **`/search`** - Pencarian tunggal dengan dukungan webhook async
- **`/search_bulk`** - Pencarian massal untuk multiple queries
- **`/health`** - Status kesehatan sistem
- **`/info`** - Informasi statistik collection
- **`/refresh`** - Force refresh collection data

---

## Metode 2: Hybrid Multi-Metric Similarity Analysis

### Deskripsi
Metode Hybrid Multi-Metric menggabungkan empat algoritma similaritas berbeda untuk menghasilkan skor akhir yang lebih akurat dan komprehensif. Metode ini menggunakan TF-IDF sebagai basis dengan kombinasi similarity metrics lainnya.

### File Terkait
- [`app.py`](app.py:1) - API endpoint dan koordinasi pencarian
- [`similarity_utils.py`](similarity_utils.py:1) - Implementasi algoritma similaritas

### Arsitektur Teknis

#### 1. Pre-computed TF-IDF Indices
```python
# Loading pre-computed indices untuk setiap kolom
for column in ['judul', 'ringkasan', 'pendahuluan', 'masalah', 'metode', 'solusi']:
    vectorizer = joblib.load(f'indices/vectorizer_{column}.pkl')
    tfidf_matrix = joblib.load(f'indices/tfidf_matrix_{column}.pkl')
```

#### 2. Four-Algorithm Approach

##### A. Jaccard Similarity (Exact Matching)
```python
def jaccard_similarity(text1, text2, n=3):
    """Jaccard similarity menggunakan word n-grams"""
    ngrams1 = ngrams(text1, n)  # Extract 3-gram kata
    ngrams2 = ngrams(text2, n)
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    return len(intersection) / len(union)
```

##### B. Levenshtein Similarity (Fuzzy Matching)
```python
def levenshtein_similarity(text1, text2):
    """Edit distance similarity untuk fuzzy matching"""
    return levenshtein_ratio(text1, text2)
```

##### C. TF-IDF Cosine Similarity
```python
def search_column(query_text, column, skema_filter=None, top_k=10):
    # 1. Preprocessing dengan Sastrawi
    processed_query = preprocess_text(query_text, stemmer, stopword_remover)
    
    # 2. Vectorization menggunakan pre-computed TF-IDF
    vectorizer = indices[column]['vectorizer']
    query_vector = vectorizer.transform([processed_query])
    
    # 3. Cosine similarity calculation
    similarity_matrix = cosine_similarity(query_vector, indices[column]['matrix'])
```

##### D. Sentence Embedding Similarity
```python
def sentence_embedding_similarity(text1, text2):
    """Semantic similarity menggunakan sentence transformers"""
    model = get_sentence_model()  # distiluse-base-multilingual-cased-v2
    emb = model.encode([text1, text2], show_progress_bar=False)
    sim = np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))
    return float(sim)
```

#### 3. Adaptive Weighted Scoring
```python
def calculate_final_score(exact, fuzzy, semantic, text1="", text2=""):
    """Algoritma scoring adaptif berdasarkan panjang teks"""
    avg_len = (len(text1) + len(text2)) / 2
    
    # Bobot adaptif berdasarkan panjang rata-rata teks
    if avg_len < 50:
        weights = (0.1, 0.3, 0.6)      # Teks pendek: prioritas semantic
    elif avg_len < 200:
        weights = (0.2, 0.4, 0.4)      # Teks sedang: seimbang fuzzy-semantic
    elif avg_len < 1000:
        weights = (0.3, 0.3, 0.4)      # Teks panjang: seimbang semua
    else:
        weights = (0.5, 0.2, 0.3)      # Teks sangat panjang: prioritas exact
    
    # Hitung skor akhir dengan bonus
    final_score = (weights[0] * exact) + (weights[1] * fuzzy) + (weights[2] * semantic)
    
    # Bonus untuk similaritas tinggi
    if semantic > 0.9: final_score += 0.05
    if fuzzy > 0.9: final_score += 0.03
    
    return min(final_score, 1.0)
```

#### 4. Indonesian Text Processing
```python
def preprocess_text(text, stemmer, stopword_remover):
    """Preprocessing khusus bahasa Indonesia dengan Sastrawi"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Hapus URL
    text = re.sub(r'\d+', '', text)                       # Hapus angka
    text = re.sub(r'[^\w\s]', '', text)                   # Hapus tanda baca
    text = stopword_remover.remove(text)                  # Hapus stopwords
    text = stemmer.stem(text)                             # Stemming
    return text.strip()
```

### Keunggulan Metode Hybrid Multi-Metric
1. **Komprehensif**: Menggabungkan 4 algoritma untuk hasil yang robust
2. **Adaptif**: Bobot yang menyesuaikan dengan karakteristik teks
3. **Akurat untuk Bahasa Indonesia**: Menggunakan Sastrawi untuk preprocessing
4. **Cepat**: Pre-computed indices untuk TF-IDF
5. **Fleksibel**: Dapat disesuaikan bobotnya sesuai kebutuhan

### Performance Features
- **Parallel Processing**: Threading untuk bulk search
- **Performance Monitoring**: Tracking waktu eksekusi dengan [`performance_monitor.py`](performance_monitor.py:1)
- **GPU Acceleration**: Dukungan MPS untuk Mac dengan AMD GPU
- **Memory Optimization**: Efficient loading of pre-computed matrices

### Endpoint API
- **`/search`** - Pencarian tunggal dengan required column parameter
- **`/search_bulk`** - Pencarian massal dengan opsi parallel processing
- **`/health`** - Status kesehatan sistem
- **`/info`** - Informasi kolom tersedia dan statistik

---

## Perbandingan Metode

| Aspek | Qdrant Vector Search | Hybrid Multi-Metric |
|-------|---------------------|---------------------|
| **Basis Teknologi** | Vector database + Sentence embeddings | TF-IDF + Multiple similarity algorithms |
| **Kecepatan** | Sangat cepat (vector search) | Cepat (pre-computed indices) |
| **Akurasi Semantic** | Tinggi (transformer model) | Sedang-Tinggi (kombinasi algorithms) |
| **Setup Complexity** | Medium (perlu Qdrant server) | Low (file-based indices) |
| **Scalability** | Sangat tinggi | Medium (terbatas memory) |
| **Resource Usage** | GPU-friendly | CPU-optimized |
| **Real-time Indexing** | Ya | Tidak (perlu rebuild indices) |
| **Language Support** | Multilingual (model-dependent) | Indonesian-optimized |

## Rekomendasi Penggunaan

### Gunakan Qdrant Method Jika:
- Dataset sangat besar (>100K dokumen)
- Perlu pencarian semantic yang sangat akurat
- Memiliki infrastruktur GPU/cloud
- Perlu real-time indexing dokumen baru
- Aplikasi web dengan traffic tinggi

### Gunakan Hybrid Multi-Metric Jika:
- Dataset kecil-medium (<100K dokumen)
- Perlu kontrol penuh atas algoritma scoring
- Infrastruktur terbatas (CPU-only)
- Perlu customization khusus bahasa Indonesia
- Aplikasi dengan requirement specific scoring

## Kesimpulan

Kedua metode memberikan solusi yang solid untuk deteksi plagiarisme dengan pendekatan yang berbeda. Qdrant method unggul dalam skalabilitas dan kecepatan semantic search, sementara Hybrid Multi-Metric method memberikan kontrol yang lebih besar dan optimasi khusus untuk bahasa Indonesia. Pilihan metode tergantung pada kebutuhan spesifik aplikasi, ukuran dataset, dan infrastruktur yang tersedia.