import pandas as pd

# Baca kedua file
proposal_baru = pd.read_csv("proposal_baru.csv")
skripsi = pd.read_csv("skripsi_with_skema.csv")

# Merge berdasarkan id
merged = pd.merge(skripsi, proposal_baru, on="id", how="left")

# Simpan hasilnya
merged.to_csv("skripsi_with_skema_merged.csv", index=False)

print("✅ Merge selesai! File disimpan ke skripsi_with_skema_merged.csv")
