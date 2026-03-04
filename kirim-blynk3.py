import requests

# Token Blynk dari kode kamu
BLYNK_TOKEN = "OHKsOsIczdPjHJ-O30Vh7GkOACHXEjCV"

# Nilai yang mau dikirim
daya = 25.76       # V0
tegangan = 11.2      # V5
arus_ACS = 2.3      # V1
statusLDR = 0       # V2 (1=terang, 0=gelap)
statusHujan = 1      # V3 (1=hujan, 0=tidak hujan)

# URL Blynk batch update
url = f"https://blynk.cloud/external/api/batch/update?token={BLYNK_TOKEN}"

# Parameter pin & value
params = {
    "V0": daya,
    "V5": tegangan,
    "V1": arus_ACS,
    "V2": statusLDR,
    "V3": statusHujan
}

# Kirim request
response = requests.get(url, params=params)

# Cek hasil
if response.status_code == 200:
    print("Data berhasil dikirim ke Blynk ✅")
else:
    print("Gagal kirim data ❌", response.status_code, response.text)
