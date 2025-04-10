# sequential_ensemble_rf_rl.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
import random

# Load dataset
df = pd.read_csv("big_output2.csv")  # features.csv should have feature columns + 'label'

# Step 1: Supervised Learning with Random Forest
df = pd.read_csv('big_output2.csv')
X = df.drop(columns=["Image", "Letters"])
y = df["Letters"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 1: Supervised Learning with Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions on training and test set
train_preds = rf.predict(X_train)
test_preds = rf.predict(X_test)

# Save predictions for test set
pd.DataFrame({"true_label": y_test.values, "predicted": test_preds}).to_csv("test_predictions.csv", index=False)

# Evaluate test accuracy
print("[Step 1] Random Forest Test Accuracy:", accuracy_score(y_test, test_preds))
print("[Step 1] Classification Report:\n", classification_report(y_test, test_preds))

# Create extended training set for RL
X_train_extended = X_train.copy()
X_train_extended["predicted_label"] = train_preds
X_train_extended["true_label"] = y_train.values

# Step 2: Reinforcement Learning (Q-Learning based toy example)

class RLAgent:
    def __init__(self, classes, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}  # key: (feature_hash), value: action (class) -> q-value
        self.classes = classes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state(self, row):
        return hash(tuple(row))

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {cls: 0.0 for cls in self.classes}
        if random.random() < self.epsilon:
            return random.choice(self.classes)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state, action, reward):
        if state not in self.q_table:
            self.q_table[state] = {cls: 0.0 for cls in self.classes}
        current_q = self.q_table[state][action]
        self.q_table[state][action] += self.alpha * (reward - current_q)

# Initialize agent
agent = RLAgent(classes=list(y.unique()))

# Train agent (one pass through training data)
for idx, row in X_train_extended.iterrows():
    features = row.drop("true_label")
    true_label = row["true_label"]
    state = agent.get_state(features)
    action = agent.choose_action(state)
    reward = 1 if action == true_label else -1
    agent.update(state, action, reward)

# Evaluate RL agent on test set
correct = 0
per_class = {label: {"correct": 0, "total": 0} for label in y.unique()}

test_X_with_preds = X_test.copy()
test_X_with_preds["predicted_label"] = test_preds  # same feature expansion

for row, true_label in zip(test_X_with_preds.iterrows(), y_test):
    _, row_data = row
    state = agent.get_state(row_data)
    action = agent.choose_action(state)
    if action == true_label:
        correct += 1
        per_class[true_label]["correct"] += 1
    per_class[true_label]["total"] += 1

rl_accuracy = correct / len(X_test)
print("\n[Step 2] RL Ensemble Accuracy on Test Set:", rl_accuracy)
print("[Step 2] Per-Class Accuracy:")
for label, stats in per_class.items():
    acc = stats["correct"] / stats["total"]
    print(f"Class {label}: {acc:.2f}")

# Optional: Cross-validation evaluation
print("\n[Cross-Validation with Random Forest only]")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
for train_idx, val_idx in skf.split(X, y):
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
    rf_cv = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_cv.fit(X_train_cv, y_train_cv)
    preds = rf_cv.predict(X_val_cv)
    acc = accuracy_score(y_val_cv, preds)
    accuracies.append(acc)
print(f"10-fold CV Accuracy (RF): Mean={np.mean(accuracies):.4f}, Std={np.std(accuracies):.4f}")
