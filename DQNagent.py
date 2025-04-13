import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv
from collections import deque

# ========== DQN Model ================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ========== DQN Agent ================
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.95, epsilon=1.0, epsilon_decay=0.75, min_epsilon=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=5000)
        self.batch_size = 64

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state, rf_suggestion=None):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(state))
        best_action = torch.argmax(q_values).item()

        if rf_suggestion is not None:
            sorted_q = torch.topk(q_values, 2).values
            if (sorted_q[0] - sorted_q[1]).item() < 0.05:
                return rf_suggestion

        return best_action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * torch.max(self.model(torch.FloatTensor(next_state))).item()
            current_q = self.model(torch.FloatTensor(state))[action]
            loss = self.loss_fn(current_q, torch.tensor(target))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# ========== Veri Hazırlığı ================
df = pd.read_csv("extended_dataset_rf.csv")
df = df.drop(columns=["Image"])

le_letters = LabelEncoder()
df["Letters"] = le_letters.fit_transform(df["Letters"])

le_rf = LabelEncoder()
df["RF_Output"] = le_rf.fit_transform(df["RF_Output"])

X = df.drop(columns=["Letters"]).values.astype(np.float32)
y = df["Letters"].values.astype(int)
rf_outputs = df["RF_Output"].values.astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ========== Ajan ve Eğitim ================
state_size = X.shape[1]
action_size = len(np.unique(y))
agent = DQNAgent(state_size, action_size)

epochs = 10
total_rewards = []

for epoch in range(epochs):
    total_reward = 0
    for i in range(len(X) - 1):
        state = X[i]
        next_state = X[i + 1]
        true_label = y[i]
        rf_suggestion = rf_outputs[i]

        action = agent.act(state, rf_suggestion)

        if action == true_label:
            reward = 10
        elif action == rf_suggestion:
            reward = 2
        else:
            reward = -1

        agent.remember(state, action, reward, next_state)
        agent.learn()
        agent.decay_epsilon()

        total_reward += reward

    total_rewards.append(total_reward)
    print(f"Epoch {epoch + 1}/{epochs} | Total reward: {total_reward}")

# ========== Doğruluk Hesaplama ================
print("\n--- Accuracy Evaluation on DQN Agent ---")

true_labels = []
agent_preds = []

for i in range(len(X)):
    state = X[i]
    true_label = y[i]
    rf_suggestion = rf_outputs[i]
    action = agent.act(state, rf_suggestion)

    true_labels.append(true_label)
    agent_preds.append(action)

# Genel doğruluk
overall_accuracy = accuracy_score(true_labels, agent_preds)
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

# Harf bazlı doğruluk hesapla
letter_accuracies = {}
for class_index in np.unique(y):
    indices = [i for i, l in enumerate(true_labels) if l == class_index]
    correct = sum(1 for i in indices if agent_preds[i] == true_labels[i])
    letter_accuracy = correct / len(indices)
    letter = le_letters.inverse_transform([class_index])[0]
    letter_accuracies[letter] = letter_accuracy
    print(f"Accuracy for '{letter}': {letter_accuracy:.4f}")

# CSV'ye yaz
with open("dqn_agent_accuracy_results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Letter", "Accuracy"])
    for letter, acc in sorted(letter_accuracies.items()):
        writer.writerow([letter, acc])
    writer.writerow([])
    writer.writerow(["Overall Accuracy", overall_accuracy])

print("\nAccuracy results saved to 'dqn_agent_accuracy_results.csv'")

# ========== Görselleştirme ================
plt.figure(figsize=(10, 5))
plt.plot(total_rewards, marker='o')
plt.title("Toplam Ödül Gelişimi")
plt.xlabel("Epoch")
plt.ylabel("Toplam Ödül")
plt.grid(True)
plt.savefig("reward_progress_dqn.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(letter_accuracies.keys(), letter_accuracies.values(), color='skyblue')
plt.title("Harf Bazlı Doğruluk")
plt.xlabel("Harf")
plt.ylabel("Doğruluk")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("letter_accuracies_dqn.png")
plt.show()
