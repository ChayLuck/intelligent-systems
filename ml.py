import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# CSV dosyasını yükle
df = pd.read_csv('big_output3.csv')
X = df.drop(columns=["Image", "Letters"]).values
y = df['Letters'].values

# Harfleri sayısal değerlere dönüştürme
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
classes = le.classes_

# Kullanılacak modeller
models = {
    "SVM": SVC(kernel="linear"),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "NaiveBayes": GaussianNB(),
    "RidgeClassifier": RidgeClassifier(),
    "Perceptron": Perceptron()
}

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Sonuçları sakla
all_results = []

# Sklearn modelleri
for name, model in models.items():
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        fold_acc = accuracy_score(y[test_idx], preds)

        row = {
            "Model": name,
            "Fold": fold + 1,
            "Average_Accuracy": round(fold_acc, 4)
        }

        for i, letter in enumerate(classes):
            true_mask = (y[test_idx] == i)
            if np.sum(true_mask) == 0:
                row[letter] = np.nan
            else:
                correct_preds = (preds[true_mask] == i)
                row[letter] = round(np.mean(correct_preds), 4)

        all_results.append(row)

# PyTorch modeli
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    model = SimpleNN(X.shape[1], len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(10):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        test_inputs = torch.tensor(X[test_idx], dtype=torch.float32)
        test_outputs = model(test_inputs)
        predicted = torch.argmax(test_outputs, dim=1).numpy()
        fold_acc = accuracy_score(y[test_idx], predicted)

        row = {
            "Model": "PyTorchNN",
            "Fold": fold + 1,
            "Average_Accuracy": round(fold_acc, 4)
        }

        for i, letter in enumerate(classes):
            true_mask = (y[test_idx] == i)
            if np.sum(true_mask) == 0:
                row[letter] = np.nan
            else:
                correct_preds = (predicted[true_mask] == i)
                row[letter] = round(np.mean(correct_preds), 4)

        all_results.append(row)

# Ortalama satırlarını ekle
results_df = pd.DataFrame(all_results)
avg_rows = []
for model_name in results_df["Model"].unique():
    model_data = results_df[results_df["Model"] == model_name]
    avg_row = {"Model": model_name, "Fold": "Avg"}
    for col in results_df.columns[2:]:
        avg_row[col] = round(model_data[col].mean(), 4)
    avg_rows.append(avg_row)

results_df = pd.concat([results_df, pd.DataFrame(avg_rows)], ignore_index=True)

# CSV'ye yaz
results_df.to_csv("full_model_report3.csv", index=False)
print("CSV oluşturuldu: full_model_report3.csv")

