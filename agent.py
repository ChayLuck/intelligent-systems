import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# ========== QLearning Agent ================
class QLearningAgent:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.9, epsilon=0.2, epsilon_decay=0.75, min_epsilon=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.weights = np.random.randn(state_size, action_size) * 0.01

    def act(self, state, rf_suggestion=None):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        q_values = np.dot(state, self.weights)
        best_action = np.argmax(q_values)

        if rf_suggestion is not None:
            second_best = np.argsort(q_values)[-2]
            gap = abs(q_values[best_action] - q_values[second_best])
            if gap < 0.05:
                return rf_suggestion

        return best_action

    def learn(self, state, action, reward, next_state):
        current_q = np.dot(state, self.weights[:, action])
        next_q = np.max(np.dot(next_state, self.weights))
        target = reward + self.gamma * next_q
        error = target - current_q

        if np.isnan(error) or np.isinf(error):
            return

        self.weights[:, action] += self.lr * error * state

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# ========== Veri Hazırlığı ================
df = pd.read_csv("extended_big_output_cv2.csv")
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
agent = QLearningAgent(state_size, action_size)

epochs = 10
reward_progress = []

for epoch in range(epochs):
    total_reward = 0
    for i in range(len(X) - 1):
        state = X[i]
        next_state = X[i + 1]
        true_label = y[i]
        rf_suggestion = rf_outputs[i]

        action = agent.act(state, rf_suggestion=rf_suggestion)

        if action == true_label:
            reward = 10
        elif action == rf_suggestion:
            reward = 2
        else:
            reward = -1

        agent.learn(state, action, reward, next_state)
        total_reward += reward
        agent.decay_epsilon()

    reward_progress.append(total_reward)
    print(f"Epoch {epoch + 1}/{epochs} | Total reward: {total_reward}")

# ========== Doğruluk Hesaplama ================
print("\n--- Accuracy Evaluation on Agent ---")

true_labels = []
agent_preds = []

for i in range(len(X)):
    state = X[i]
    true_label = y[i]
    rf_suggestion = rf_outputs[i]
    action = agent.act(state, rf_suggestion=rf_suggestion)

    true_labels.append(true_label)
    agent_preds.append(action)

overall_accuracy = accuracy_score(true_labels, agent_preds)
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

letter_accuracies = {}
for class_index in np.unique(y):
    indices = [i for i, l in enumerate(true_labels) if l == class_index]
    correct = sum(1 for i in indices if agent_preds[i] == true_labels[i])
    letter_accuracy = correct / len(indices)
    letter = le_letters.inverse_transform([class_index])[0]
    letter_accuracies[letter] = letter_accuracy
    print(f"Accuracy for '{letter}': {letter_accuracy:.4f}")

with open("agent_accuracy_results_new2.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Letter", "Accuracy"])
    for letter, acc in sorted(letter_accuracies.items()):
        writer.writerow([letter, acc])
    writer.writerow([])
    writer.writerow(["Overall Accuracy", overall_accuracy])

print("\nAccuracy results saved to 'agent_accuracy_results_new2.csv'")

# ========== Karışıklık Matrisi ================
cm = confusion_matrix(true_labels, agent_preds)
plt.figure(figsize=(18, 14))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=le_letters.classes_, yticklabels=le_letters.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix of Ensemble Q-Learning Agent")
plt.tight_layout()
plt.savefig("confusion_matrix6.png")
plt.show()

# ========== Eğitim Süreci Grafiği ================
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), reward_progress, marker='o', linestyle='-')
plt.title("Q-Learning Agent Training Progress")
plt.xlabel("Epoch")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_progress_qlearning2.png")
plt.show()

# ========== Harf Bazlı Doğruluk Grafiği ================
plt.figure(figsize=(12, 6))
sorted_items = sorted(letter_accuracies.items())
letters = [item[0] for item in sorted_items]
accuracies = [item[1] for item in sorted_items]

bars = plt.bar(letters, accuracies, color="skyblue")
plt.ylim(0, 1)
plt.title("Letter-wise Accuracy of Q-Learning Agent")
plt.xlabel("Letters")
plt.ylabel("Accuracy")
plt.grid(axis='y')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f"{height:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("letter_accuracies_qlearning2.png")
plt.show()

# ========== Hatalı Tahminler ================
with open("misclassified_letters4.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Index", "True Label", "Predicted Label"])
    for idx, (true, pred) in enumerate(zip(true_labels, agent_preds)):
        if true != pred:
            writer.writerow([idx, le_letters.inverse_transform([true])[0], le_letters.inverse_transform([pred])[0]])

print("Misclassified letters saved to 'misclassified_letters4.csv'")
