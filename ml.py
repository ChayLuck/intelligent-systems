import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# CSV dosyasını yükle
df = pd.read_csv('big_output2.csv')
X = df.drop(columns=["Image", "Letters"]).values  # Görüntüden elde edilen özellikler
y = df['Letters'].values  # Gerçek harfler

# Harfleri sayısal değerlere dönüştürme
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Kullanılacak modeller
models = {
    "SVM": SVC(),
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

# Cross-validation ayarları
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Sonuçları saklayacak liste
results = []

# Sklearn Modelleri İçin Cross-Validation
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    for fold, score in enumerate(scores):
        results.append([name, fold + 1, score])

# PyTorch Modeli
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
    model = SimpleNN(X.shape[1], len(le.classes_))
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
        predicted = torch.argmax(test_outputs, dim=1)
        acc = (predicted == torch.tensor(y[test_idx])).float().mean().item()
    
    results.append(["PyTorchNN", fold + 1, acc])

# Sonuçları CSV'ye kaydetme
results_df = pd.DataFrame(results, columns=['Model', 'Fold', 'Accuracy'])
results_df.to_csv('model_comparison_results.csv', index=False)
