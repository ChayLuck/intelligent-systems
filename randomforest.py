import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Veriyi oku ve X, y olarak ayır
df = pd.read_csv('big_output2.csv')
X = df.drop(columns=["Image", "Letters"]).values
y = df["Letters"].values

# Harfleri sayıya çevir
le = LabelEncoder()
y = le.fit_transform(y)
classes = le.classes_

# Stratified 10-fold için setup
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
all_results = []

# Her harf için doğruluk hesaplama fonksiyonu
def per_class_accuracy(y_true, y_pred, classes):
    result = {}
    for i, label in enumerate(classes):
        mask = (y_true == i)
        if np.sum(mask) == 0:
            result[label] = np.nan
        else:
            result[label] = round(np.mean(y_pred[mask] == i), 4)
    return result

# Sadece Random Forest modeli ile eğitim ve test
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    model = RandomForestClassifier()
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    acc = accuracy_score(y[test_idx], preds)

    row = {"Model": "RandomForest", "Fold": fold + 1, "Average_Accuracy": round(acc, 4)}
    row.update(per_class_accuracy(y[test_idx], preds, classes))
    all_results.append(row)

# Ortalama değerleri hesapla
results_df = pd.DataFrame(all_results)
avg_row = {"Model": "RandomForest", "Fold": "Avg"}
for col in results_df.columns[2:]:
    avg_row[col] = round(results_df[col].mean(), 4)
results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)

# CSV dosyasına yaz
results_df.to_csv("random_forest_report2.csv", index=False)
print("CSV oluşturuldu: random_forest_report2.csv")
