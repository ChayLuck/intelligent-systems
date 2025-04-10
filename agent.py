import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
from sklearn.metrics import accuracy_score
import csv

# ========== QLearning Agent ================
class QLearningAgent:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.9, epsilon=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.weights = np.random.randn(state_size, action_size) * 0.01  # Küçük başlangıç

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        q_values = np.dot(state, self.weights)
        return np.argmax(q_values)

    def learn(self, state, action, reward, next_state):
        current_q = np.dot(state, self.weights[:, action])
        next_q = np.max(np.dot(next_state, self.weights))
        target = reward + self.gamma * next_q
        error = target - current_q

        if np.isnan(error) or np.isinf(error):
            return  # Skip update

        self.weights[:, action] += self.lr * error * state

# ========== Veri Hazırlığı ================
df = pd.read_csv("extended_dataset_rf.csv")
df = df.drop(columns=["Image"])

le_letters = LabelEncoder()
df["Letters"] = le_letters.fit_transform(df["Letters"])

le_rf = LabelEncoder()
df["RF_Output"] = le_rf.fit_transform(df["RF_Output"])

X = df.drop(columns=["Letters"]).values.astype(np.float32)
y = df["Letters"].values.astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ========== Ajan ve Eğitim ================
state_size = X.shape[1]
action_size = len(np.unique(y))
agent = QLearningAgent(state_size, action_size)

epochs = 10
for epoch in range(epochs):
    total_reward = 0
    for i in range(len(X) - 1):
        state = X[i]
        next_state = X[i + 1]
        true_label = y[i]

        action = agent.act(state)
        reward = 10 if action == true_label else -1
        agent.learn(state, action, reward, next_state)
        total_reward += reward

    print(f"Epoch {epoch+1}/{epochs} | Total reward: {total_reward}")




#///////////////////////////////////////////
from sklearn.metrics import accuracy_score
import csv

# Tüm tahminleri ve gerçek etiketleri topla
true_labels = []
agent_preds = []

for i in range(len(X)):
    state = np.array(X[i], dtype=np.float32)
    true_label = y[i]
    action = agent.act(state)
    predicted_label = le_letters.inverse_transform([action])[0]

    true_labels.append(true_label)
    agent_preds.append(predicted_label)

# Genel doğruluk
overall_accuracy = accuracy_score(true_labels, agent_preds)
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

# Harf bazlı doğruluk hesapla
unique_labels = sorted(set(true_labels))
letter_accuracies = {}

for letter in unique_labels:
    indices = [i for i, l in enumerate(true_labels) if l == letter]
    correct = sum(1 for i in indices if agent_preds[i] == true_labels[i])
    letter_accuracy = correct / len(indices)
    letter_accuracies[letter] = letter_accuracy
    print(f"Accuracy for '{letter}': {letter_accuracy:.4f}")

# CSV'ye yaz
with open("agent_accuracy_results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Letter", "Accuracy"])
    for letter, acc in letter_accuracies.items():
        writer.writerow([letter, acc])
    writer.writerow([])
    writer.writerow(["Overall Accuracy", overall_accuracy])

print("\nAccuracy results saved to 'agent_accuracy_results.csv'")



    
