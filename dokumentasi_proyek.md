# BAB 4 HASIL DAN PEMBAHASAN

Bab ini menyajikan hasil implementasi dari sistem deteksi plagiarisme yang telah dikembangkan. Pembahasan akan difokuskan pada dua komponen perangkat lunak utama, yaitu `app.py` sebagai server aplikasi dan `similarity_utils.py` sebagai modul inti untuk perhitungan kemiripan. Selain itu, akan diuraikan pula alur kerja sistem secara keseluruhan.

## 4.1 Implementasi Server Aplikasi (`app.py`)

Server aplikasi, yang diimplementasikan dalam file `app.py`, merupakan pusat kendali dari sistem ini. Dibangun menggunakan framework Flask, server ini bertanggung jawab untuk menangani seluruh permintaan API, mengelola proses pencarian, dan mengembalikan hasil kepada pengguna.

### 4.1.1 Inisialisasi dan Pemuatan Model

Sebelum server siap menerima permintaan, proses inisialisasi yang krusial dijalankan melalui fungsi `init_app()`. Tahapan ini memastikan bahwa semua sumber daya yang diperlukan telah dimuat ke dalam memori untuk operasi yang efisien.
- **Pustaka Bahasa**: Sistem menginisialisasi _stemmer_ dan _stopword remover_ dari library `Sastrawi` untuk memproses teks berbahasa Indonesia.
- **Data dan Indeks**: Metadata dari `indices/metadata.csv` dan indeks TF-IDF dari file `.pkl` yang sesuai dimuat. Indeks ini mencakup berbagai bagian proposal seperti 'judul', 'ringkasan', dan 'metode', yang memungkinkan pencarian berbasis kata kunci yang sangat cepat.
- **Model Semantik**: Model _Sentence Transformer_ (`distiluse-base-multilingual-cased-v2`) dimuat melalui pemanggilan `initialize_sentence_model()` untuk analisis kemiripan makna.

### 4.1.2 Pra-pemrosesan Teks

Setiap teks yang masuk, baik dari permintaan pengguna maupun dari database, dinormalisasi melalui fungsi `preprocess_text`. Proses ini mencakup:
1.  Konversi ke huruf kecil.
2.  Penghapusan URL, angka, dan tanda baca.
3.  Penghapusan _stopwords_ (kata umum).
4.  _Stemming_ (pengubahan kata ke bentuk dasar).
Langkah ini penting untuk memastikan perbandingan yang adil dan akurat.

### 4.1.3 Endpoint API

Fungsionalitas sistem diekspos melalui beberapa endpoint RESTful API:
-   **`POST /search`**: Melayani permintaan pencarian untuk satu teks. Endpoint ini mendukung mode sinkron (hasil langsung dikembalikan) dan asinkron (menggunakan webhook).
-   **`POST /search_bulk`**: Dirancang untuk menangani pencarian dalam jumlah besar (batch). Endpoint ini juga mendukung mode sinkron dan asinkron, dengan opsi pemrosesan paralel (`use_parallel`) untuk meningkatkan throughput.
-   **`GET /health`** dan **`GET /info`**: Endpoint utilitas untuk memantau status server dan mendapatkan informasi mengenai data yang tersedia.

### 4.1.4 Logika Pencarian Inti (`search_column`)

Fungsi ini merupakan mesin utama di balik proses pencarian.
1.  **Penyaringan Awal (TF-IDF)**: Menggunakan _cosine similarity_ pada vektor TF-IDF untuk secara cepat mengidentifikasi dokumen-dokumen yang paling relevan dari database.
2.  **Analisis Lanjutan**: Untuk dokumen-dokumen yang lolos penyaringan awal, sistem melakukan perhitungan skor yang lebih canggih menggunakan metrik dari `similarity_utils.py`, yaitu Jaccard, Levenshtein, dan kemiripan semantik (embedding).
3.  **Skor Akhir**: Semua skor tersebut diagregasi menjadi `final_score` melalui fungsi `calculate_final_score` yang menggunakan pembobotan adaptif.
4.  **Hasil**: Mengembalikan daftar hasil yang terurut berdasarkan `final_score`.

## 4.2 Implementasi Utilitas Similaritas (`similarity_utils.py`)

Modul `similarity_utils.py` adalah kumpulan perangkat lunak yang berisi fungsi-fungsi untuk mengukur kemiripan antara dua teks dengan berbagai pendekatan.

### 4.2.1 Metode Perhitungan Similaritas

Tiga metode utama digunakan untuk menghasilkan skor kemiripan yang komprehensif:
-   **`jaccard_similarity`**: Menghitung kemiripan berdasarkan perbandingan himpunan n-gram kata. Metode ini efektif untuk mendeteksi kesamaan frasa secara harfiah.
-   **`levenshtein_similarity`**: Mengukur "jarak suntingan" antar teks. Metode ini toleran terhadap kesalahan ketik atau perbedaan kecil pada level karakter.
-   **`sentence_embedding_similarity`**: Metode paling canggih yang mengukur kemiripan makna (semantik). Teks diubah menjadi vektor numerik menggunakan model _Sentence Transformer_, lalu kemiripannya dihitung. Ini memungkinkan sistem untuk memahami bahwa "mobil" dan "kendaraan roda empat" memiliki makna yang serupa.

### 4.2.2 Perhitungan Skor Akhir (`calculate_final_score`)

Fungsi ini secara cerdas menggabungkan ketiga skor di atas. Dengan menggunakan sistem pembobotan adaptif berdasarkan panjang teks, fungsi ini dapat memberikan penekanan yang berbeda pada setiap metrik. Misalnya, untuk teks yang sangat panjang, kemiripan eksak (Jaccard) mungkin lebih diutamakan, sedangkan untuk teks pendek, kemiripan semantik menjadi lebih penting.

## 4.3 Alur Kerja Sistem

Secara ringkas, alur kerja sistem saat menerima permintaan deteksi plagiarisme adalah sebagai berikut:

1.  **Input**: Klien mengirimkan permintaan ke endpoint `/search` atau `/search_bulk`.
2.  **Normalisasi**: Teks input diproses melalui `preprocess_text`.
3.  **Pencarian TF-IDF**: Pencarian cepat dilakukan untuk menyaring dokumen kandidat.
4.  **Analisis Mendalam**: Skor Jaccard, Levenshtein, dan Semantik dihitung untuk kandidat teratas.
5.  **Agregasi Skor**: `calculate_final_score` menggabungkan skor-skor tersebut.
6.  **Output**: Hasil akhir yang telah diurutkan dikembalikan kepada klien dalam format JSON.