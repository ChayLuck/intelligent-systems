import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# ====== Parameters ======
EPOCHS = 75
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 3.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.85
LR = 0.0005

# ====== Dataset Preparation ======
df = pd.read_csv("extended_big_output_cv5.csv")
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
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
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

            reward = 10 if action == labels[j].item() else (3 if action == rf_preds[j].item() else -3)
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
