import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

data = {
    'jumlah_transaksi': [100, 120, 130, 90, 5000, 100, 110, 95, 105, 150, 200, 100, 95, 3000],
    'frekuensi_transaksi': [1, 1, 2, 1, 10, 1, 1, 1, 2, 1, 1, 1, 1, 20],
    'waktu_transaksi': [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2]
}

df = pd.DataFrame(data)

print("Data Transaksi:")
print(df)

X = df[['jumlah_transaksi', 'frekuensi_transaksi', 'waktu_transaksi']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(contamination=0.2, random_state=42)
model.fit(X_scaled)

predictions = model.predict(X_scaled)

df['prediksi'] = predictions
df['status'] = df['prediksi'].apply(lambda x: 'Normal' if x == 1 else 'Anomali')

print("\nHasil Deteksi Anomali:")
print(df)

def deteksi_penipuan(jumlah_transaksi, frekuensi_transaksi, waktu_transaksi):
    input_data = np.array([[jumlah_transaksi, frekuensi_transaksi, waktu_transaksi]])
    input_scaled = scaler.transform(input_data)
    
    prediksi = model.predict(input_scaled)
    
    if prediksi == 1:
        return "Transaksi Normal"
    else:
        return "Transaksi Mencurigakan (Anomali)"

print("\nMasukkan data transaksi untuk deteksi penipuan:")
jumlah_transaksi = float(input("Jumlah Transaksi: "))
frekuensi_transaksi = int(input("Frekuensi Transaksi: "))
waktu_transaksi = int(input("Waktu Transaksi (1 untuk normal, 2 untuk anomali): "))

hasil = deteksi_penipuan(jumlah_transaksi, frekuensi_transaksi, waktu_transaksi)
print(f"Hasil Prediksi: {hasil}")
