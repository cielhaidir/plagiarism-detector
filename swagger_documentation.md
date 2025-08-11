# API Documentation - Swagger/OpenAPI Specification

## Server 1: TF-IDF Based Plagiarism Detection API (Port 5000)

```yaml
openapi: 3.0.3
info:
  title: TF-IDF Plagiarism Detection API
  description: |
    API untuk deteksi plagiarisme menggunakan TF-IDF, Jaccard similarity, Levenshtein similarity, dan sentence embeddings.
    Server ini menggunakan pendekatan hybrid dengan multiple similarity metrics untuk hasil yang lebih akurat.
  version: 1.0.0
  contact:
    name: Plagiarism Detection Team
    email: support@plagiarism-api.com
servers:
  - url: http://localhost:5000
    description: Development server (TF-IDF based)
  - url: https://api-tfidf.plagiarism-detection.com
    description: Production server (TF-IDF based)

paths:
  /search:
    post:
      summary: Single text plagiarism search
      description: |
        Melakukan pencarian plagiarisme untuk satu teks dengan menggunakan TF-IDF sebagai filter awal,
        kemudian menghitung multiple similarity metrics (Jaccard, Levenshtein, Semantic).
      tags:
        - Search
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - query_text
                - column
              properties:
                query_text:
                  type: string
                  description: Teks yang akan dicari kemiripannya
                  example: "Sistem informasi berbasis web untuk manajemen data mahasiswa"
                column:
                  type: string
                  enum: [judul, ringkasan, pendahuluan, masalah, metode, solusi]
                  description: Kolom proposal yang akan dicari
                  example: "judul"
                skema:
                  type: string
                  description: Filter berdasarkan skema proposal (opsional)
                  example: "PKM-KC"
                top_k:
                  type: integer
                  default: 10
                  minimum: 1
                  maximum: 100
                  description: Jumlah hasil teratas yang dikembalikan
                  example: 10
                webhook_url:
                  type: string
                  format: uri
                  description: URL webhook untuk pemrosesan asinkron (opsional)
                  example: "https://your-app.com/webhook/plagiarism-results"
      responses:
        '200':
          description: Hasil pencarian (sinkron)
          content:
            application/json:
              schema:
                type: object
                properties:
                  results:
                    type: array
                    items:
                      $ref: '#/components/schemas/SearchResult'
                  query_info:
                    $ref: '#/components/schemas/QueryInfo'
        '202':
          description: Pencarian dimulai (asinkron)
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id:
                    type: string
                    format: uuid
                    example: "550e8400-e29b-41d4-a716-446655440000"
                  status:
                    type: string
                    example: "processing"
                  message:
                    type: string
                    example: "Search started. Results will be sent to webhook when complete."
                  webhook_url:
                    type: string
                    format: uri
        '400':
          description: Bad request
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '500':
          description: Internal server error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /search_bulk:
    post:
      summary: Bulk plagiarism search
      description: |
        Melakukan pencarian plagiarisme untuk multiple teks sekaligus.
        Mendukung pemrosesan paralel dan sequential.
      tags:
        - Search
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - texts
              properties:
                texts:
                  type: array
                  items:
                    type: object
                    properties:
                      proposal_id:
                        type: string
                        description: ID unik untuk proposal ini
                        example: "PROP-001"
                      skema:
                        type: string
                        description: Skema proposal
                        example: "PKM-KC"
                      judul:
                        type: string
                        example: "Sistem Manajemen Perpustakaan Digital"
                      ringkasan:
                        type: string
                        example: "Penelitian ini mengembangkan sistem..."
                      pendahuluan:
                        type: string
                      masalah:
                        type: string
                      metode:
                        type: string
                      solusi:
                        type: string
                  description: Array objek teks yang akan dicari
                top_k:
                  type: integer
                  default: 1
                  minimum: 1
                  maximum: 50
                  description: Jumlah hasil teratas per query
                use_parallel:
                  type: boolean
                  default: true
                  description: Gunakan pemrosesan paralel untuk performa lebih baik
                max_workers:
                  type: integer
                  description: Jumlah worker threads (opsional, default = CPU cores)
                  example: 4
                webhook_url:
                  type: string
                  format: uri
                  description: URL webhook untuk pemrosesan asinkron (opsional)
      responses:
        '200':
          description: Hasil bulk search (sinkron)
          content:
            application/json:
              schema:
                type: object
                properties:
                  bulk_results:
                    type: array
                    items:
                      $ref: '#/components/schemas/BulkSearchItem'
                  total_queries:
                    type: integer
                  processing_method:
                    type: string
                    enum: [parallel, sequential]
                  max_workers:
                    type: integer
                    nullable: true
        '202':
          description: Bulk search dimulai (asinkron)
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id:
                    type: string
                    format: uuid
                  status:
                    type: string
                    example: "processing"
                  message:
                    type: string
                  processing_method:
                    type: string
                    enum: [parallel, sequential]

  /health:
    get:
      summary: Health check
      description: Memeriksa status kesehatan server dan apakah indeks telah dimuat
      tags:
        - System
      responses:
        '200':
          description: Server status
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: "healthy"
                  indices_loaded:
                    type: boolean
                    example: true

  /info:
    get:
      summary: System information
      description: Mendapatkan informasi tentang kolom yang tersedia dan statistik database
      tags:
        - System
      responses:
        '200':
          description: System information
          content:
            application/json:
              schema:
                type: object
                properties:
                  available_columns:
                    type: array
                    items:
                      type: string
                    example: ["judul", "ringkasan", "pendahuluan", "masalah", "metode", "solusi"]
                  total_proposals:
                    type: integer
                    example: 15000
                  unique_skemas:
                    type: array
                    items:
                      type: string
                    example: ["PKM-KC", "PKM-K", "PKM-M", "PKM-T"]

components:
  schemas:
    SearchResult:
      type: object
      properties:
        id:
          type: integer
          description: ID proposal yang cocok
          example: 12345
        skema:
          type: string
          description: Skema proposal
          example: "PKM-KC"
        similarity_score:
          type: number
          format: float
          description: Skor TF-IDF similarity (legacy)
          example: 0.85
        exact_score:
          type: number
          format: float
          description: Jaccard similarity score
          example: 0.72
        fuzzy_score:
          type: number
          format: float
          description: Levenshtein similarity score
          example: 0.89
        semantic_score:
          type: number
          format: float
          description: Sentence embedding similarity score
          example: 0.91
        final_score:
          type: number
          format: float
          description: Combined weighted similarity score
          example: 0.87
        column:
          type: string
          description: Kolom yang dicari
          example: "judul"
        matched_text:
          type: string
          description: Teks asli yang cocok dari database
          example: "Sistem Informasi Manajemen Data Mahasiswa Berbasis Web"

    BulkSearchItem:
      type: object
      properties:
        query_index:
          type: integer
          description: Index query dalam array input
          example: 0
        proposal_id:
          type: string
          description: ID proposal dari input
          example: "PROP-001"
        results:
          type: array
          items:
            $ref: '#/components/schemas/SearchResult'
        query_info:
          type: object
          properties:
            skema_filter:
              type: string
              nullable: true
            columns_searched:
              type: array
              items:
                type: string
            total_results:
              type: integer

    QueryInfo:
      type: object
      properties:
        column:
          type: string
        skema_filter:
          type: string
          nullable: true
        total_results:
          type: integer

    Error:
      type: object
      properties:
        error:
          type: string
          description: Error message
          example: "query_text is required"

tags:
  - name: Search
    description: Operasi pencarian plagiarisme
  - name: System
    description: Informasi sistem dan health check
```

---

## Server 2: Qdrant Vector-Based Plagiarism Detection API (Port 5001)

```yaml
openapi: 3.0.3
info:
  title: Qdrant Vector-Based Plagiarism Detection API
  description: |
    API untuk deteksi plagiarisme menggunakan Qdrant vector database.
    Server ini menggunakan sentence embeddings untuk pencarian semantik yang cepat dan akurat.
  version: 1.0.0
  contact:
    name: Plagiarism Detection Team
    email: support@plagiarism-api.com
servers:
  - url: http://localhost:5001
    description: Development server (Qdrant based)
  - url: https://api-qdrant.plagiarism-detection.com
    description: Production server (Qdrant based)

paths:
  /search:
    post:
      summary: Fast vector similarity search
      description: |
        Melakukan pencarian plagiarisme menggunakan vector similarity dengan Qdrant.
        Lebih cepat daripada TF-IDF untuk pencarian semantik.
      tags:
        - Search
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - query_text
              properties:
                query_text:
                  type: string
                  description: Teks yang akan dicari kemiripannya
                  example: "Sistem informasi berbasis web untuk manajemen data mahasiswa"
                column:
                  type: string
                  enum: [judul, ringkasan, pendahuluan, masalah, metode, solusi]
                  description: Kolom proposal yang akan dicari (opsional)
                  example: "judul"
                skema:
                  type: string
                  description: Filter berdasarkan skema proposal (opsional)
                  example: "PKM-KC"
                top_k:
                  type: integer
                  default: 10
                  minimum: 1
                  maximum: 100
                  description: Jumlah hasil teratas yang dikembalikan
                threshold:
                  type: number
                  format: float
                  default: 0.7
                  minimum: 0.0
                  maximum: 1.0
                  description: Threshold minimum similarity score
                  example: 0.7
                webhook_url:
                  type: string
                  format: uri
                  description: URL webhook untuk pemrosesan asinkron (opsional)
      responses:
        '200':
          description: Hasil pencarian (sinkron)
          content:
            application/json:
              schema:
                type: object
                properties:
                  results:
                    type: array
                    items:
                      $ref: '#/components/schemas/QdrantSearchResult'
                  query_info:
                    $ref: '#/components/schemas/QdrantQueryInfo'
        '202':
          description: Pencarian dimulai (asinkron)
        '503':
          description: System is initializing
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /search_bulk:
    post:
      summary: Fast bulk vector search
      description: Melakukan pencarian plagiarisme untuk multiple teks menggunakan Qdrant
      tags:
        - Search
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - texts
              properties:
                texts:
                  type: array
                  items:
                    type: object
                    properties:
                      proposal_id:
                        type: string
                        example: "PROP-001"
                      judul:
                        type: string
                      ringkasan:
                        type: string
                      pendahuluan:
                        type: string
                      masalah:
                        type: string
                      metode:
                        type: string
                      solusi:
                        type: string
                top_k:
                  type: integer
                  default: 5
                  description: Jumlah hasil teratas per query
                threshold:
                  type: number
                  format: float
                  default: 0.7
                  description: Threshold minimum similarity score
                webhook_url:
                  type: string
                  format: uri
                  description: URL webhook untuk pemrosesan asinkron (opsional)
      responses:
        '200':
          description: Hasil bulk search (sinkron)
          content:
            application/json:
              schema:
                type: object
                properties:
                  bulk_results:
                    type: array
                    items:
                      $ref: '#/components/schemas/QdrantBulkSearchItem'
                  total_queries:
                    type: integer
        '202':
          description: Bulk search dimulai (asinkron)

  /health:
    get:
      summary: Health check
      description: Memeriksa status kesehatan server Qdrant
      tags:
        - System
      responses:
        '200':
          description: Server status
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    enum: [healthy, initializing]
                    example: "healthy"
                  qdrant_ready:
                    type: boolean
                    example: true
                  initialization_complete:
                    type: boolean
                    example: true

  /info:
    get:
      summary: System information
      description: Mendapatkan informasi tentang Qdrant dan statistik
      tags:
        - System
      responses:
        '200':
          description: System information
          content:
            application/json:
              schema:
                type: object
                properties:
                  search_engine:
                    type: string
                    example: "Qdrant"
                  available_columns:
                    type: array
                    items:
                      type: string
                  stats:
                    $ref: '#/components/schemas/QdrantStats'
        '503':
          description: System not ready

  /stats:
    get:
      summary: Detailed statistics
      description: Mendapatkan statistik detail dari Qdrant
      tags:
        - System
      responses:
        '200':
          description: Detailed statistics
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/QdrantStats'

  /refresh:
    get:
      summary: Force refresh Qdrant collection
      description: Memaksa reinisialisasi koleksi Qdrant
      tags:
        - System
      responses:
        '200':
          description: Refresh initiated
          content:
            application/json:
              schema:
                type: object
                properties:
                  message:
                    type: string
                    example: "Qdrant collection refresh initiated"
                  status:
                    type: string
                    example: "refreshing"

  /index_proposals:
    post:
      summary: Yearly data indexing with year filtering
      description: |
        Endpoint untuk mengindeks proposal baru dengan filter tahun (2025+).
        Mendukung pemrosesan sinkron dan asinkron dengan webhook support.
        Otomatis menolak proposal dengan tahun < 2025.
      tags:
        - Indexing
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - proposals
              properties:
                proposals:
                  type: array
                  items:
                    type: object
                    required:
                      - id
                      - judul
                      - skema
                      - tahun
                    properties:
                      id:
                        type: integer
                        description: ID unik proposal
                        example: 99991
                      judul:
                        type: string
                        description: Judul proposal
                        example: "Penerapan Machine Learning untuk Deteksi Plagiarisme"
                      skema:
                        type: string
                        description: Skema proposal
                        example: "Penelitian Dasar"
                      tahun:
                        type: integer
                        description: Tahun proposal (harus >= 2025)
                        example: 2025
                      ringkasan:
                        type: string
                        description: Ringkasan proposal
                      pendahuluan:
                        type: string
                        description: Pendahuluan proposal
                      masalah:
                        type: string
                        description: Masalah yang diangkat
                      metode:
                        type: string
                        description: Metode yang digunakan
                      solusi:
                        type: string
                        description: Solusi yang ditawarkan
                webhook_url:
                  type: string
                  format: uri
                  description: URL webhook untuk pemrosesan asinkron
                  example: "https://your-app.com/webhook/indexing-results"
                year_threshold:
                  type: integer
                  default: 2025
                  minimum: 2025
                  description: Tahun minimum untuk proposal yang diterima
                  example: 2025
      responses:
        '200':
          description: Indexing completed successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: "success"
                  message:
                    type: string
                    example: "Successfully indexed 2 proposals"
                  indexed_count:
                    type: integer
                    example: 2
                  year_threshold:
                    type: integer
                    example: 2025
        '202':
          description: Indexing started (asynchronous)
          content:
            application/json:
              schema:
                type: object
                  properties:
                    job_id:
                      type: string
                      format: uuid
                      example: "550e8400-e29b-41d4-a716-446655440000"
                    status:
                      type: string
                      example: "processing"
                    message:
                      type: string
                      example: "Indexing started. Results will be sent to webhook when complete."
                    year_threshold:
                      type: integer
                      example: 2025
        '400':
          description: Bad request - validation error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '500':
          description: Internal server error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

components:
  schemas:
    QdrantSearchResult:
      type: object
      properties:
        id:
          type: integer
          description: ID proposal yang cocok
        similarity_score:
          type: number
          format: float
          description: Vector similarity score
          example: 0.89
        column:
          type: string
          description: Kolom yang cocok
        skema:
          type: string
          description: Skema proposal
        matched_text:
          type: string
          description: Teks yang cocok
        text:
          type: string
          description: Teks lengkap (alias untuk matched_text)
        judul:
          type: string
          description: Judul proposal (jika tersedia)

    QdrantBulkSearchItem:
      type: object
      properties:
        query_index:
          type: integer
          description: Index query dalam array input
        proposal_id:
          type: string
          description: ID proposal dari input
        results:
          type: array
          items:
            $ref: '#/components/schemas/QdrantSearchResult'

    QdrantQueryInfo:
      type: object
      properties:
        column:
          type: string
          nullable: true
        skema_filter:
          type: string
          nullable: true
        total_results:
          type: integer
        threshold:
          type: number
          format: float

    QdrantStats:
      type: object
      properties:
        collection_name:
          type: string
          example: "plagiarism_vectors"
        total_vectors:
          type: integer
          example: 50000
        vector_size:
          type: integer
          example: 512
        indexed_points:
          type: integer
        status:
          type: string

    Error:
      type: object
      properties:
        error:
          type: string
          description: Error message

tags:
  - name: Search
    description: Operasi pencarian plagiarisme menggunakan Qdrant
  - name: System
    description: Informasi sistem dan kontrol Qdrant
```

---

## Perbandingan Kedua Server

| Aspek | TF-IDF Server (Port 5000) | Qdrant Server (Port 5001) |
|-------|---------------------------|---------------------------|
| **Metode Pencarian** | TF-IDF + Multiple Similarity Metrics | Vector Similarity (Embeddings) |
| **Kecepatan** | Sedang (karena multiple calculations) | Cepat (optimized vector search) |
| **Akurasi** | Tinggi (hybrid approach) | Tinggi (semantic understanding) |
| **Resource Usage** | CPU intensive | GPU/CPU optimized |
| **Use Case** | Analisis detail dengan multiple metrics | Pencarian cepat dengan fokus semantik |
| **Response Format** | Multiple similarity scores | Single similarity score |

## Contoh Penggunaan

### TF-IDF Server (Port 5000)
```bash
# Single search
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "Sistem manajemen perpustakaan digital",
    "column": "judul",
    "top_k": 5
  }'

# Bulk search
curl -X POST http://localhost:5000/search_bulk \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      {
        "proposal_id": "PROP-001",
        "judul": "Sistem Informasi Akademik",
        "skema": "PKM-KC"
      }
    ],
    "top_k": 3,
    "use_parallel": true
  }'
```

## Contoh Penggunaan /index_proposals

### Indexing Proposal Baru (2025+)
```bash
# Flask Server (Port 5000)
curl -X POST http://localhost:5000/index_proposals \
  -H "Content-Type: application/json" \
  -d '{
    "proposals": [
      {
        "id": 99991,
        "judul": "Penerapan Machine Learning untuk Deteksi Plagiarisme",
        "skema": "Penelitian Dasar",
        "tahun": 2025,
        "ringkasan": "Penelitian ini mengembangkan sistem deteksi plagiarisme menggunakan ML",
        "pendahuluan": "Plagiarisme merupakan masalah serius dalam dunia akademik",
        "masalah": "Bagaimana meningkatkan akurasi deteksi plagiarisme?",
        "metode": "Menggunakan algoritma ensemble learning",
        "solusi": "Sistem deteksi plagiarisme berbasis ML dengan akurasi 95%"
      }
    ],
    "year_threshold": 2025
  }'

# Qdrant Server (Port 5001)
curl -X POST http://localhost:5001/index_proposals \
  -H "Content-Type: application/json" \
  -d '{
    "proposals": [
      {
        "id": 99992,
        "judul": "Analisis Sentimen pada Review Produk E-commerce",
        "skema": "Penelitian Terapan",
        "tahun": 2025,
        "ringkasan": "Menganalisis sentimen pelanggan terhadap produk e-commerce",
        "pendahuluan": "Review produk menjadi sumber informasi penting",
        "masalah": "Bagaimana meningkatkan akurasi analisis sentimen?",
        "metode": "Implementasi LSTM dan transformer models",
        "solusi": "Model analisis sentimen dengan akurasi 92%"
      }
    ]
  }'

# Asynchronous indexing with webhook
curl -X POST http://localhost:5000/index_proposals \
  -H "Content-Type: application/json" \
  -d '{
    "proposals": [...],
    "webhook_url": "https://your-app.com/webhook/indexing-results"
  }'
```

### Response Examples

**Success Response (200):**
```json
{
  "status": "success",
  "message": "Successfully indexed 2 proposals",
  "indexed_count": 2,
  "year_threshold": 2025,
  "filtered_count": 0,
  "total_processed": 2
}
```

**Asynchronous Response (202):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "message": "Indexing started. Results will be sent to webhook when complete.",
  "year_threshold": 2025
}
```

**Validation Error (400):**
```json
{
  "error": "Validation failed",
  "details": [
    "Proposal at index 0 missing required fields: ['skema']",
    "Proposal at index 1 tahun must be >= 2025"
  ]
}
```

**Webhook Payload Example:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "timestamp": "2025-08-11T02:30:00Z",
  "result": {
    "status": "success",
    "message": "Successfully indexed 2 proposals",
    "indexed_count": 2,
    "year_threshold": 2025
  }
}
```

## Error Handling

### Year Filtering
- **2025+**: Accepted for indexing
- **2024 and below**: Automatically rejected with appropriate message

### Validation Requirements
- All required fields must be present: `id`, `judul`, `skema`, `tahun`
- `tahun` must be >= `year_threshold` (default: 2025)
- All text fields should be strings
- `id` must be unique (duplicate IDs will be skipped)

### Common Error Responses
- **400 Bad Request**: Validation errors, missing required fields
- **500 Internal Server Error**: Server-side processing errors
- **503 Service Unavailable**: System is initializing

### Qdrant Server (Port 5001)
```bash
# Single search
curl -X POST http://localhost:5001/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "Sistem manajemen perpustakaan digital",
    "threshold": 0.8,
    "top_k": 5
  }'

# Bulk search
curl -X POST http://localhost:5001/search_bulk \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      {
        "proposal_id": "PROP-001",
        "judul": "Sistem Informasi Akademik"
      }
    ],
    "threshold": 0.7
  }'