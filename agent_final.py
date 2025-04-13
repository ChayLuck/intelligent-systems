import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# ====== Parameters ======
EPOCHS = 100
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 4.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.65
LR = 0.001

# ====== Dataset Preparation ======
df = pd.read_csv("extended_big_output_cv6.csv")
df = df.drop(columns=["Image"])

le_letters = LabelEncoder()
df["Letters"] = le_letters.fit_transform(df["Letters"])

le_rf = LabelEncoder()
df["RF_Output"] = le_rf.fit_transform(df["RF_Output"])

X = df.drop(columns=["Letters"]).values.astype(np.float32)
y = df["Letters"].values.astype(int)
rf_out = df["RF_Output"].values.astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X)
y = torch.tensor(y)
rf_out = torch.tensor(rf_out)

n_classes = len(np.unique(y))

# ====== Neural Network (Improved DQN) ======
class ImprovedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImprovedDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ====== Agent Setup ======
model = ImprovedDQN(X.shape[1], n_classes)
target_model = ImprovedDQN(X.shape[1], n_classes)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
epsilon = EPSILON_START

# ====== Tracking metrics for visualization ======
history = {
    'epoch': [],
    'loss': [],
    'reward': [],
    'epsilon': []
}

# ====== Training ======
for epoch in range(EPOCHS):
    indices = torch.randperm(X.size(0))
    total_loss = 0
    total_reward = 0

    for i in range(0, X.size(0), BATCH_SIZE):
        batch_idx = indices[i:i+BATCH_SIZE]
        states = X[batch_idx]
        labels = y[batch_idx]
        rf_preds = rf_out[batch_idx]

        q_values = model(states)
        with torch.no_grad():
            next_q_values = target_model(states)

        targets = q_values.clone().detach()

        for j in range(states.size(0)):
            if random.random() < epsilon:
                action = random.randint(0, n_classes - 1)
            else:
                action = torch.argmax(q_values[j]).item()

            reward = 8 if action == labels[j].item() else (1 if action == rf_preds[j].item() else -5)
            total_reward += reward

            max_next_q = torch.max(next_q_values[j]).item()
            targets[j, action] = reward + GAMMA * max_next_q

        loss = criterion(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0:
        target_model.load_state_dict(model.state_dict())

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Total Reward: {total_reward}")
    
    # Store metrics for visualization
    history['epoch'].append(epoch + 1)
    history['loss'].append(total_loss)
    history['reward'].append(total_reward)
    history['epsilon'].append(epsilon)

# ====== Evaluation ======
model.eval()
preds = []
true_labels = []

with torch.no_grad():
    for i in range(X.size(0)):
        q_vals = model(X[i])
        pred = torch.argmax(q_vals).item()
        preds.append(pred)
        true_labels.append(y[i].item())

acc = accuracy_score(true_labels, preds)
print(f"\nFinal Accuracy of Improved DQN Agent: {acc:.4f}")

# Optional: Letter-wise Accuracy
letter_acc = {}
true_labels = np.array(true_labels)
preds = np.array(preds)
for label in np.unique(true_labels):
    mask = true_labels == label
    correct = (preds[mask] == true_labels[mask]).sum()
    letter_acc[le_letters.inverse_transform([label])[0]] = correct / mask.sum()

print("\nLetter-wise Accuracies:")
for letter, acc in sorted(letter_acc.items()):
    print(f"{letter}: {acc:.4f}")

# ====== Visualizations ======
plt.style.use('ggplot')

# 1. Training metrics over epochs
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

# Plot loss
axes[0].plot(history['epoch'], history['loss'], 'b-')
axes[0].set_title('Loss Over Training', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True)

# Plot reward
axes[1].plot(history['epoch'], history['reward'], 'g-')
axes[1].set_title('Total Reward Over Training', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Total Reward')
axes[1].grid(True)

# Plot epsilon decay
axes[2].plot(history['epoch'], history['epsilon'], 'r-')
axes[2].set_title('Epsilon Decay Over Training', fontsize=14)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Epsilon')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.close()

# 2. Letter-wise accuracy visualization
letters = list(letter_acc.keys())
accs = list(letter_acc.values())

plt.figure(figsize=(12, 8))
bars = plt.bar(letters, accs, color='skyblue')
plt.axhline(y=acc, color='r', linestyle='-', label=f'Overall Accuracy: {acc:.4f}')
plt.title('Letter-wise Recognition Accuracy', fontsize=16)
plt.xlabel('Letters')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1.1)

# Add accuracy values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom', rotation=0)

plt.legend()
plt.tight_layout()
plt.savefig('letter_accuracy.png')
plt.close()

# 3. Confusion Matrix
cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_letters.inverse_transform(np.unique(true_labels)),
            yticklabels=le_letters.inverse_transform(np.unique(true_labels)))
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# 4. Epsilon vs Accuracy Trade-off
# For this we need to simulate predictions at different epsilon values
epsilon_range = np.linspace(0, 1, 10)
accuracy_at_epsilon = []

model.eval()
for eps in epsilon_range:
    current_preds = []
    for i in range(X.size(0)):
        if random.random() < eps:
            # Random action
            pred = random.randint(0, n_classes - 1)
        else:
            # Greedy action
            with torch.no_grad():
                q_vals = model(X[i])
                pred = torch.argmax(q_vals).item()
        current_preds.append(pred)
    
    current_acc = accuracy_score(y.numpy(), current_preds)
    accuracy_at_epsilon.append(current_acc)

plt.figure(figsize=(10, 6))
plt.plot(epsilon_range, accuracy_at_epsilon, 'bo-')
plt.title('Accuracy vs Epsilon Trade-off', fontsize=14)
plt.xlabel('Epsilon (Randomness)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig('epsilon_accuracy_tradeoff.png')
plt.close()

print("\nVisualization results saved as PNG files.")