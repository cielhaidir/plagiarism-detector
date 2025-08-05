**Rencana Implementasi: Menyimpan Hasil Preprocessing Teks untuk Efisiensi Pencarian**

---

### Tujuan

Menghindari proses preprocessing ulang terhadap matched\_text saat pencarian, dengan cara menyimpan hasil preprocessing setiap kolom saat indexing.

---

### Tahapan Implementasi

#### 1. **Modifikasi Script Indexing (`create_indices`)**

Tambahkan penyimpanan hasil `processed_texts` ke dalam file CSV per kolom.

```python
# Setelah membuat processed_texts
processed_output_path = os.path.join(output_dir, f'processed_{column}.csv')
processed_texts.to_csv(processed_output_path, index=False, header=False)
```

#### 2. **Modifikasi Script Pencarian (`search_column`)**

Tambahkan cache loader untuk hasil preprocessing dari file yang sudah disimpan.

```python
processed_texts_cache = {}

def get_preprocessed_matched(proposal_id, column):
    if column not in processed_texts_cache:
        path = f"indices/processed_{column}.csv"
        processed_texts_cache[column] = pd.read_csv(path, header=None)[0]
    return processed_texts_cache[column].iloc[proposal_id]
```

Ganti bagian:

```python
pre_m = preprocess_text(matched_text, stemmer, stopword_remover)
```

menjadi:

```python
pre_m = get_preprocessed_matched(proposal_id, column)
```

---

### Keuntungan

* Proses pencarian lebih cepat karena tidak melakukan preprocessing ulang.
* Konsistensi hasil preprocessing antara indexing dan pencarian.

### Risiko

* Butuh penyimpanan tambahan untuk file CSV hasil preprocessing.
* Jika preprocessing logic berubah, file hasil lama bisa tidak relevan.

### Mitigasi Risiko

* Tambahkan versioning/preprocessing hash di nama file jika logika berubah.
* Validasi data secara berkala.

---

### Checklist Implementasi

* [ ] Tambahkan ekspor `processed_texts` ke CSV saat indexing
* [ ] Tambahkan loader hasil preprocessing di fungsi pencarian
* [ ] Lakukan pengujian perbandingan skor sebelum dan sesudah perubahan
* [ ] Dokumentasi perubahan di README / dokumentasi teknis
