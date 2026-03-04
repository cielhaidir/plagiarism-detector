import requests

# Token Blynk dari kode kamu
BLYNK_TOKEN = "OHKsOsIczdPjHJ-O30Vh7GkOACHXEjCV"

# Nilai yang mau dikirim
daya = 32.1       # V0
tegangan = 12.1   # V5
arus_ACS = 2.6    # V1
statusLDR = 0     # V2 (1=terang, 0=gelap)
statusHujan = 1   # V3 (1=hujan, 0=tidak hujan)

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

# Kirim data sensor
response = requests.get(url, params=params)

if response.status_code == 200:
    print("Data berhasil dikirim ke Blynk ✅")
else:
    print("Gagal kirim data ❌", response.status_code, response.text)

# # --- BYPASS SIMULASI NOTIF PANEL BERSIH ---
# event_url = "https://blynk.cloud/external/api/logEvent"
# event_params = {
#     "token": BLYNK_TOKEN,
#     "code": "panel_bersih",
#     "description": "Panel surya sudah kembali bersih."
# }
# r_event = requests.get(event_url, params=event_params)

# if r_event.status_code == 200:
#     print("Notifikasi panel bersih terkirim ✅")
# else:
#     print("Gagal kirim notifikasi ❌", r_event.text)
