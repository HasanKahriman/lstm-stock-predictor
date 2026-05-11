import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.metrics import RootMeanSquaredError 

# Veri Okuma
borsaDf = pd.read_csv("stock_data.csv")
borsaDf["Date"] = pd.to_datetime(borsaDf["Date"])

# Fiyat verisini scaler için 2 boyutlu yapma
fiyat = borsaDf["Close"].values.reshape(-1, 1)

# Veriyi Bölme
train,test=train_test_split(fiyat,test_size=0.2,shuffle=False)

# Normalizasyon
scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test) 

# Ölçeklendirilmiş veri üzerinden sequence oluşturma
# Sliding Window Mantığı
def sequence_olustur(veri, adim=30):
    x, y = [], []
    for f in range(len(veri) - adim):
        x.append(veri[f : f+adim])
        y.append(veri[f+adim])
    return np.array(x), np.array(y)
# Test ve Trainleri belirleme işlemi
x_train, y_train = sequence_olustur(train_scaled, 30)
x_test, y_test = sequence_olustur(test_scaled, 30)

# Model Kurulumu
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(30, 1)))
model.add(Dropout(0.2))#Modelimizin ezber yapmasını engeller
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(50))
model.add(Dense(1))

# Modeli Derleme
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae", RootMeanSquaredError(name="rmse")])

# Modeli eğitme
#validation_data parametresini yine modelin ezber yapmasını önlemek için kullanıyoruz.
model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_test, y_test))

# Modeli Kaydetme
model.save("model.keras")
print("Model başarıyla eğitildi ve kaydedildi!")