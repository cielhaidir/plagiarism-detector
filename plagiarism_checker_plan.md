# Plan: Plagiarism Detection Service with API

This document outlines the plan for building a plagiarism detection service that includes a real-time API.

## Key Requirement: Language

The entire implementation, especially text processing and cleaning, will be specifically tailored for the **Indonesian language**. This includes using Indonesian stop words and tokenization rules.

## Architecture Overview

The system is designed in two main parts: an offline **Indexing** process to prepare the data and an online **API Server** to handle real-time plagiarism checks.

1.  **Phase 1: Indexing (A one-time setup process)**
    *   The large `skripsi_with_skema.csv` file will be processed once to create an efficient, searchable "index." This index will contain pre-processed text and its corresponding numerical vector representation. This process runs at the start of the application, and the resulting index is saved to disk to avoid re-running it on every launch.

2.  **Phase 2: API Server (The live service)**
    *   A lightweight web server (using Flask for Python) will serve the API.
    *   On startup, the server will load the pre-built index into memory for fast querying.
    *   It will expose `/search` and `/search_bulk` endpoints to allow users to check for plagiarism via API calls.

## Workflow Diagram

```mermaid
graph TD
    subgraph "Phase 1: Indexing (Offline/Startup)"
        A[Start: skripsi_with_skema.csv] --> B[Run Indexer];
        B --> C[Preprocess & Vectorize All Proposals];
        C --> D[Save Index (Vectors & Metadata) to Disk];
    end

    subgraph "Phase 2: API Server (Online)"
        E[API Server Starts] --> F[Load Index from Disk];
        F --> G{API Endpoints};
        G -- "/search (POST)" --> H[Handle Single Query];
        G -- "/search_bulk (POST)" --> I[Handle Bulk Query];

        H --> J[Process Query & Calculate Similarity];
        I --> J;

        J --> K[Return Top Matches as JSON];
    end

    subgraph "User"
        L[User/Application] -- "Sends HTTP Request" --> G;
        K -- "Receives JSON Response" --> L;
    end
```

## Detailed Step-by-Step Breakdown

### Phase 0: Building the Plagiarism Detection Engine

*   **Goal:** To implement sophisticated algorithms and techniques for accurate plagiarism detection in Indonesian text, providing detailed insights into plagiarism types (exact, fuzzy, semantic).
*   **Technology:** Python with NLTK, Scikit-learn, Gensim (for Word2Vec/FastText), custom algorithms, and potentially a library for Levenshtein distance (e.g., `python-Levenshtein`).
*   **Steps:**

1.  **Indonesian Text Preprocessing Module:**
    *   **Normalization:** Handle various Indonesian spellings, remove diacritics, standardize punctuation.
    *   **Tokenization:** Split text into words using Indonesian-specific rules.
    *   **Stop Words Removal:** Create and use comprehensive Indonesian stop words list (dan, di, yang, untuk, dengan, pada, adalah, ini, itu, etc.).
    *   **Stemming:** Implement Indonesian stemming using algorithms like Nazief-Adriani or Sastrawi library.
    *   **N-gram Generation:** Generate word and character n-grams (e.g., 1-gram, 2-gram, 3-gram) for different levels of matching.

2.  **Multiple Similarity Metrics and Detailed Plagiarism Detection:**
    *   **Exact Matching (N-gram based):**
        *   **N-gram Overlap:** Calculate the percentage of overlapping n-grams between texts.
        *   **Jaccard Similarity:** Compute the Jaccard index based on the set of unique n-grams. This will provide an "exact" match score.
    *   **Fuzzy Matching (Levenshtein Distance):**
        *   **Sentence-level Levenshtein:** Measure the edit distance between potentially rephrased sentences. This will provide a "fuzzy" match score.
    *   **Semantic Matching (TF-IDF & Word Embeddings):**
        *   **TF-IDF Cosine Similarity:** Continue using TF-IDF for document-level semantic similarity.
        *   **Sentence Embeddings Comparison:** Utilize pre-trained Indonesian word embeddings (e.g., Word2Vec, FastText) or sentence transformers to generate vector representations of sentences/paragraphs. Compare these vectors using cosine similarity for deeper semantic understanding. This will provide a "semantic" match score.

3.  **Weighted Scoring and Result Granularity:**
    *   **Individual Score Calculation:** For each potential plagiarism instance, calculate separate `exact_score`, `fuzzy_score`, and `semantic_score`.
    *   **Combined Plagiarism Score:** Implement a weighted scoring algorithm as requested:
        ```python
        def calculate_final_score(exact_score, fuzzy_score, semantic_score):
            # Example weights, tunable based on desired emphasis
            return (0.4 * exact_score) + (0.3 * fuzzy_score) + (0.3 * semantic_score)
        ```
    *   **Highlighting Plagiarized Text:** The output should include not just scores, but also identify *which parts* of the text (e.g., specific sentences or paragraphs) are plagiarized, and potentially highlight them. This implies storing the original text segments and their corresponding matches.

4.  **Advanced Detection Techniques & Context Handling:**
    *   **Chunk-based Analysis:** Break documents into smaller, manageable chunks (paragraphs, sentences) for localized and more precise plagiarism detection. This helps in pinpointing plagiarized sections.
    *   **Citation Analysis:** Develop logic to identify and potentially exclude properly cited content from the `daftar_pustaka` field (if provided and formatted consistently) to reduce false positives.
    *   **Pattern Recognition:** Implement rules or machine learning models to detect common plagiarism patterns like sentence reordering, synonym replacement, or structural similarities.

5.  **Thresholding and Confidence:**
    *   **Dynamic Thresholds:** Set different similarity thresholds based on the type of match (exact, fuzzy, semantic) and the document length/type.
    *   **False Positive Reduction:** Implement filters and heuristics to reduce false positives (e.g., common phrases, boilerplate text).
    *   **Confidence Scoring:** Provide confidence levels for each plagiarism detection, indicating the likelihood of actual plagiarism.

6.  **Performance Optimization (Re-evaluation):**
    *   **Vectorization & Embedding Caching:** Cache generated vectors and embeddings to speed up comparisons.
    *   **Approximate Nearest Neighbors (ANN):** For large datasets, use ANN algorithms (e.g., Faiss, Annoy) for faster similarity searches, especially for semantic matching.
    *   **Parallel Processing:** Utilize multi-threading or multiprocessing for bulk comparisons and computationally intensive tasks.

```mermaid
graph TD
    A[Input Text] --> B{Indonesian Text Preprocessing}
    B --> C[Tokenization & N-gram Generation]
    B --> D[Stemming]
    
    C --> E[Exact Match (Jaccard, N-gram Overlap)]
    D --> F[Fuzzy Match (Levenshtein Distance)]
    D --> G[Semantic Match (TF-IDF Cosine)]
    C --> H[Semantic Match (Sentence Embeddings)]
    
    E --> I[Individual Scores]
    F --> I
    G --> I
    H --> I
    
    I --> J{Weighted Score Combination}
    J --> K[Final Plagiarism Score]
    K --> L[Highlight Plagiarized Text]
    
    subgraph "Plagiarism Detection Engine"
        A
        B
        C
        D
        E
        F
        G
        H
        I
        J
        K
        L
    end
```

</search>

### Phase 1: Indexing

*   **Goal:** To create a fast, searchable database of all proposals from `skripsi_with_skema.csv`.
*   **Technology:** Python with Pandas and Scikit-learn.
*   **Steps:**
    1.  **Stream and Process:** Read `skripsi_with_skema.csv` row by row to manage memory usage.
    2.  **Clean and Combine:** For each row, combine and clean the text from all relevant columns (`judul`, `ringkasan`, etc.) using Indonesian language rules.
    3.  **Vectorize with TF-IDF:** Convert the cleaned text into TF-IDF vectors, using a vocabulary and stop word list appropriate for Indonesian.
    4.  **Save Index Files:** Save the trained TF-IDF vectorizer model, the resulting vectors, and a mapping back to the original proposal IDs and schemas. This will likely create a few files on disk (e.g., `tfidf_model.pkl`, `proposal_vectors.npz`, `proposal_metadata.json`).

### Phase 2: API Server

*   **Goal:** To provide a stable, fast API for plagiarism detection.
*   **Technology:** Python with Flask.
*   **Steps:**
    1.  **Setup Flask App:** Create a simple Flask web application.
    2.  **Load Index on Startup:** When the app starts, it will load the files generated in Phase 1.
    3.  **Implement `/search` Endpoint:**
        *   Define a function for the `/search` route that accepts `POST` requests.
        *   Parse the incoming JSON to get `query_text`, `column`, and optional `skema`.
        *   Preprocess the `query_text` using the same Indonesian-specific method as the indexer.
        *   Transform the query text into a vector using the loaded TF-IDF model.
        *   If `skema` is provided, filter the main index to only compare against proposals with that schema.
        *   Calculate the cosine similarity between the query vector and the indexed vectors.
        *   Return a JSON list of the top matching proposals, including their ID, schema, and similarity score.
    4.  **Implement `/search_bulk` Endpoint:**
        *   Define a function for the `/search_bulk` route.
        *   Parse the incoming JSON to get the list of `texts`.
        *   Loop through each text object in the list.
        *   For each object, perform the same steps as the single `/search` endpoint (process, vectorize, filter, compare).
        *   Collect the results for all objects and return them in a structured JSON response.
