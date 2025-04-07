import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Take the csv and split it into X and y
df = pd.read_csv('big_output10.csv')
X = df.drop(columns=["Image", "Letters"]).values
y = df["Letters"].values

# Turn lettters into numbers and classes save the letters
le = LabelEncoder()
y = le.fit_transform(y)
classes = le.classes_

# Models
models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='liblinear'),
    "DecisionTree": DecisionTreeClassifier(),
    "NaiveBayes": GaussianNB(),
    "RidgeClassifier": RidgeClassifier(),
    "Perceptron": Perceptron()
}
# Split the data into 10 folds and shuffle it with a random state
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
all_results = []

# Helper: Katman bazında doğruluk hesaplama / Per-class accuracy calculation for letters
def per_class_accuracy(y_true, y_pred, classes):
    result = {}
    for i, label in enumerate(classes):
        mask = (y_true == i)
        if np.sum(mask) == 0:
            result[label] = np.nan
        else:
            result[label] = round(np.mean(y_pred[mask] == i), 4)
    return result

# Training and evaluation for sklearn models
for name, model in models.items():
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], preds)

        row = {"Model": name, "Fold": fold + 1, "Average_Accuracy": round(acc, 4)}
        row.update(per_class_accuracy(y[test_idx], preds, classes))
        all_results.append(row)

# PyTorch modeli (Simple neural network with 128 neurons)
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Training and evaluation with PyTorch
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    model = SimpleNN(X.shape[1], len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train_tensor = torch.tensor(y[train_idx], dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(10):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X[test_idx], dtype=torch.float32)
        outputs = model(X_test_tensor)
        preds = torch.argmax(outputs, dim=1).numpy()
        acc = accuracy_score(y[test_idx], preds)

        row = {"Model": "PyTorchNN", "Fold": fold + 1, "Average_Accuracy": round(acc, 4)}
        row.update(per_class_accuracy(y[test_idx], preds, classes))
        all_results.append(row)

# Calculate average values of each model in 10 folds
results_df = pd.DataFrame(all_results)
avg_rows = []

for model_name in results_df["Model"].unique():
    model_data = results_df[results_df["Model"] == model_name]
    avg_row = {"Model": model_name, "Fold": "Avg"}
    for col in results_df.columns[2:]:
        avg_row[col] = round(model_data[col].mean(), 4)
    avg_rows.append(avg_row)

results_df = pd.concat([results_df, pd.DataFrame(avg_rows)], ignore_index=True)

# Save
results_df.to_csv("full_model_report10.csv", index=False)
print("CSV oluşturuldu: full_model_report10.csv")
