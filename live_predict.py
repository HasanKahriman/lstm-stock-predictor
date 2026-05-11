import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
# Modeli yükleme
model=load_model("model.keras")
print("Model başarıyla yüklendi.")
# Ölçeklendirmede doğru değerleri almak için önceki değerlerimizle eğitiyoruz
borsaDf = pd.read_csv("stock_data.csv")
fiyatlar = borsaDf["Close"].values.reshape(-1, 1)
split_index = int(len(fiyatlar) * 0.8)
train_veri = fiyatlar[:split_index]
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_veri)

# Her gün %0.5 ile %1.5 arası artan 30 günlük simülasyon verisi
simulasyon_veri = [10000] # Başlangıç fiyatı
for _ in range(29):
    yeni_fiyat = simulasyon_veri[-1] * (1 + np.random.uniform(0.005, 0.015))
    simulasyon_veri.append(yeni_fiyat)
# 2 boyutlu hale getirme
simulasyon_veri = [[f] for f in simulasyon_veri]

print("Simülasyon Başlıyor. Her 1 saniyede yeni bir fiyat üretilecek.")
print("----------------------------------------------------------------------------------------------------------------------")
# Canlı grafik kurulumu
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))

# İlk ürettiğimiz 30 günü grafikte göstermek için
gecmis_gun_indeksleri = list(range(1, len(simulasyon_veri) + 1))
gecmis_fiyat_serisi = [f[0] for f in simulasyon_veri]
gun_indeksleri = []
mevcut_fiyat_serisi = []
tahmin_serisi = []
sinyal_serisi = []
yuzde_degisim_serisi = []

try:
    gun = 1
    while gun<=60:
        #Normalize etme
        son_30_gün = np.array(simulasyon_veri[-30:]).reshape(-1, 1)
        son_30_scaled = scaler.transform(son_30_gün)
        x = son_30_scaled.reshape(1, 30, 1)

        # Predict
        tahmin_scaled = model.predict(x, verbose=0)
        # Tekrar bizim anlayacagımız formata dondurme işlemi
        tahmin_gercek = scaler.inverse_transform(tahmin_scaled)[0][0]

        # Terminale yazdırma
        su_an_ki_fiyat = simulasyon_veri[-1][0]
        print(f"Mevcut Fiyat: {su_an_ki_fiyat:.2f} TL | Modelin Yarın İçin Fiyat Beklentisi: {tahmin_gercek:.2f} TL")
        # Sinyal için yuzde hesaplama
        yuzde_degisim=((tahmin_gercek-su_an_ki_fiyat)/su_an_ki_fiyat)*100
        #Sinyal türü belirleme
        if yuzde_degisim>1.0:
            sinyal="AL"
            sinyal_renk="green"
        elif yuzde_degisim<-1.0:
            sinyal="SAT"
            sinyal_renk="red"
        else:
            sinyal="BEKLE"
            sinyal_renk="orange"
        print(f"Yüzde Değişim: {yuzde_degisim:.2f}%.{sinyal}!!!")
        
        # Canlı grafik değerlerini güncelleme
        gun_indeksleri.append(gun)
        mevcut_fiyat_serisi.append(su_an_ki_fiyat)
        tahmin_serisi.append(tahmin_gercek)
        sinyal_serisi.append(sinyal)
        yuzde_degisim_serisi.append(yuzde_degisim)

        ax.clear()
        # Geçmiş başlangıç verisi
        ax.plot(gecmis_gun_indeksleri,gecmis_fiyat_serisi,color="gray",linewidth=1.2,alpha=0.6, label="İlk 30 günün verisi")
        # Günlük canlı fiyat grafiği
        canli_baslangic = len(simulasyon_veri) - len(mevcut_fiyat_serisi)
        canli_gun_indeksleri = list(range(canli_baslangic, canli_baslangic + len(mevcut_fiyat_serisi)))
        ax.plot(canli_gun_indeksleri, mevcut_fiyat_serisi, marker="o", linewidth=1.5, label="Mevcut Fiyat")
        # Tahmin Grafiği
        ax.plot(canli_gun_indeksleri,tahmin_serisi,marker="x", linewidth=1.5,linestyle="--",label="Model Tahmini")
        # Gün gün sinyal noktaları(Enumaerate ile hem indexi hem de sinyali alıyoruz.)
        al_x = [canli_gun_indeksleri[i] for i, s in enumerate(sinyal_serisi) if s == "AL"]
        al_y = [mevcut_fiyat_serisi[i] for i, s in enumerate(sinyal_serisi) if s == "AL"]
        sat_x = [canli_gun_indeksleri[i] for i, s in enumerate(sinyal_serisi) if s == "SAT"]
        sat_y = [mevcut_fiyat_serisi[i] for i, s in enumerate(sinyal_serisi) if s == "SAT"]
        bekle_x = [canli_gun_indeksleri[i] for i, s in enumerate(sinyal_serisi) if s == "BEKLE"]
        bekle_y = [mevcut_fiyat_serisi[i] for i, s in enumerate(sinyal_serisi) if s == "BEKLE"]
        # Sinyali grafiğe yazdırma
        if al_x:
            ax.scatter(al_x, al_y, color="green", marker="^", s=70, label="AL Sinyali")
        if sat_x:
            ax.scatter(sat_x, sat_y, color="red", marker="v", s=70, label="SAT Sinyali")
        if bekle_x:
            ax.scatter(bekle_x, bekle_y, color="orange", marker="o", s=45, label="BEKLE Sinyali")
        ax.set_title("Güncellenen Günlük Fiyat ve Sonraki Gün için Fiyat Tahmini")
        ax.set_xlabel("Gün")
        ax.set_ylabel("Fiyat")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.pause(0.01)
        
        ax.text(
            0.02, 0.98,
            f"Sinyal: {sinyal}\nBeklenen Değişim: %{yuzde_degisim:.2f}",
            transform=ax.transAxes,fontsize=11,verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor=sinyal_renk)
        )
    
        # Hayali fiyat üretme işlemi
        degisim = 1 + np.random.uniform(-0.005, 0.005)
        yeni_fiyat = su_an_ki_fiyat * degisim

        simulasyon_veri.append([yeni_fiyat])
        gun += 1

        time.sleep(1)

except KeyboardInterrupt:
    print("Simülasyon durduruldu.")
finally:
    plt.ioff()
    plt.show()


