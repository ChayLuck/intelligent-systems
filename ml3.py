import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# CSV dosyasını oku ve X, y olarak ayır
df = pd.read_csv('big_output10.csv')
X = df.drop(columns=["Image", "Letters"]).values
y = df["Letters"].values

# Harfleri sayılara dönüştür ve sınıfları sakla
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_

# Verileri 10 kata böl
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Harf başına doğruluk hesaplama yardımcı fonksiyonu
def per_class_accuracy(y_true, y_pred, classes):
    result = {}
    for i, label in enumerate(classes):
        mask = (y_true == i)
        if np.sum(mask) == 0:
            result[label] = np.nan
        else:
            result[label] = round(np.mean(y_pred[mask] == i), 4)
    return result

# Cross-validation sonuçlarını depolamak için
cv_results = []
fold_accuracies = []

# Tüm verilerin tahminlerini depolamak için dizi
all_predictions = np.zeros_like(y_encoded)
all_indices_covered = np.zeros_like(y_encoded, dtype=bool)

print("Cross-validation ile RandomForest modeli eğitiliyor ve tahminler yapılıyor...")

# Her kata için model eğitimi ve test seti tahmini
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y_encoded)):
    model = RandomForestClassifier()
    model.fit(X[train_idx], y_encoded[train_idx])
    
    # Test verisi üzerinde tahmin
    preds = model.predict(X[test_idx])
    acc = accuracy_score(y_encoded[test_idx], preds)
    
    # Cross-validation sonuçlarını kaydet
    row = {"Model": "RandomForest", "Fold": fold + 1, "Average_Accuracy": round(acc, 4)}
    row.update(per_class_accuracy(y_encoded[test_idx], preds, classes))
    cv_results.append(row)
    fold_accuracies.append(acc)
    
    # Her test örneği için tahmini sakla
    all_predictions[test_idx] = preds
    all_indices_covered[test_idx] = True
    
    print(f"Fold {fold+1}/10 tamamlandı. Doğruluk: {acc:.4f}")

# Tüm örneklerin tahmin edildiğinden emin ol
if not np.all(all_indices_covered):
    print("Uyarı: Bazı örnekler hiçbir test setinde yer almadı!")

# Tahminleri harflere dönüştür
predictions_letters = le.inverse_transform(all_predictions)

# Cross-validation genel sonuçları
avg_row = {"Model": "RandomForest", "Fold": "Avg"}
result_df = pd.DataFrame(cv_results)
for col in result_df.columns[2:]:
    avg_row[col] = round(result_df[col].mean(), 4)
cv_results.append(avg_row)

# Her harf için doğruluk sonuçları
for letter in classes:
    letter_idx = np.where(le.transform([letter])[0] == y_encoded)[0]
    correct = np.sum(all_predictions[letter_idx] == y_encoded[letter_idx])
    total = len(letter_idx)
    accuracy = correct / total if total > 0 else 0
    print(f"{letter} için doğruluk: {accuracy:.4f} ({correct}/{total})")

# Orijinal veri çerçevesine cross-validation tahminlerini ekle
df['RF_Output'] = predictions_letters

# Genişletilmiş veri çerçevesini kaydet
df.to_csv("extended_big_output_cv10.csv", index=False)
print(f"CSV oluşturuldu: extended_big_output_cv2.csv")

# Cross-validation sonuçlarını kaydet
pd.DataFrame(cv_results).to_csv("rf_model_report_cv10.csv", index=False)
print(f"Model raporu oluşturuldu: rf_model_report_cv2.csv")
print(f"10-katlı cross-validation ortalama doğruluk: {np.mean(fold_accuracies):.4f}")