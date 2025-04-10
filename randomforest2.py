import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Veriyi oku
df = pd.read_csv('big_output2.csv')
X = df.drop(columns=["Image", "Letters"]).values
y = df["Letters"].values

# Harfleri sayıya çevir
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Random Forest modeliyle tüm veriyi eğit
rf_model = RandomForestClassifier()
rf_model.fit(X, y_encoded)

# Modelle tahmin yap
rf_preds = rf_model.predict(X)

# Tahminleri orijinal sınıf etiketlerine çevir (opsiyonel ama okunabilirlik için güzel olur)
rf_preds_labels = le.inverse_transform(rf_preds)

# Yeni sütun olarak ekle
df["RF_Output"] = rf_preds_labels  # veya istersen rf_preds (sayı) da olur

# Yeni veri kümesini kaydet
df.to_csv("extended_dataset_rf.csv", index=False)
print("Yeni genişletilmiş veri kümesi oluşturuldu: extended_dataset_rf.csv")
