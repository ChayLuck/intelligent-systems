import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# ====== Parameters ======

EPOCHS = 100 #how many time
BATCH_SIZE = 128 #how many data
GAMMA = 0.70 #how much stick to rewards
EPSILON_START = 4.0 #randomeness start point, exploration rate
EPSILON_END = 0.2 #randomeness end point
EPSILON_DECAY = 0.85 #randomeness will drop over time
LR = 0.001 #learning rate
FOLDS = 10 #how many folds for cross validation

# ====== Dataset Reading and Prep ======
df = pd.read_csv("extended_big_output_cv6.csv")
df = df.drop(columns=["Image"])

le_letters = LabelEncoder()
df["Letters"] = le_letters.fit_transform(df["Letters"])
#Turn these values to encoded numbers so we can work on
le_rf = LabelEncoder()
df["RF_Output"] = le_rf.fit_transform(df["RF_Output"])

X = df.drop(columns=["Letters"]).values.astype(np.float32)
y = df["Letters"].values.astype(int)
rf_out = df["RF_Output"].values.astype(int)

#Scaler makes the values between 0 and 1, so agent wont confuse the importance
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Torch need the data in tensor format (similar to arrays)
X = torch.tensor(X)
y = torch.tensor(y)
rf_out = torch.tensor(rf_out)

n_classes = len(np.unique(y))

# ====== Neural Network (Improved DQN) ====== (Dropout for reduce overfitting),(Bigger values means deeper network)
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

# ====== K-Fold Cross Validation ======
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold_accuracies = []
fold_letter_accuracies = []
history_all_folds = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\n=== Fold {fold+1}/{FOLDS} ===")

    X_train, y_train, rf_train = X[train_idx], y[train_idx], rf_out[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # ====== Agent Setup ======
    model = ImprovedDQN(X.shape[1], n_classes)
    target_model = ImprovedDQN(X.shape[1], n_classes)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=LR) #update the weights for reducing the loss
    criterion = nn.MSELoss() #this one using regression, calculating the loss
    epsilon = EPSILON_START

    history = {
        'epoch': [],
        'loss': [],
        'reward': [],
        'epsilon': []
    }

    # ====== Training Loop ======
    for epoch in range(EPOCHS):
        indices = torch.randperm(X_train.size(0))
        total_loss = 0
        total_reward = 0

        for i in range(0, X_train.size(0), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            states = X_train[batch_idx]
            labels = y_train[batch_idx]
            rf_preds = rf_train[batch_idx]

            q_values = model(states)
            with torch.no_grad():
                next_q_values = target_model(states)

            targets = q_values.clone().detach()
            #if epsilon high it will possibly pick random action
            for j in range(states.size(0)):
                if random.random() < epsilon:
                    action = random.randint(0, n_classes - 1)
                else:
                    action = torch.argmax(q_values[j]).item()

                reward = 8 if action == labels[j].item() else (-3 if action == rf_preds[j].item() else -5)
                total_reward += reward

                #Bellman equation for Q-learning
                max_next_q = torch.max(next_q_values[j]).item()
                targets[j, action] = reward + GAMMA * max_next_q

            #clear the gradients and update the weights
            loss = criterion(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        #update the target model every 5 epochs
        if epoch % 5 == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        history['epoch'].append(epoch + 1)
        history['loss'].append(total_loss)
        history['reward'].append(total_reward)
        history['epsilon'].append(epsilon)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Total Reward: {total_reward:.2f}")

    # ====== Evaluation ======
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for i in range(X_test.size(0)):
            q_vals = model(X_test[i])
            pred = torch.argmax(q_vals).item()
            preds.append(pred)
            true_labels.append(y_test[i].item())

    acc = accuracy_score(true_labels, preds)
    fold_accuracies.append(acc)

    print(f"Fold {fold+1} Accuracy: {acc:.4f}")

    # Letter-wise accuracy
    letter_acc = {}
    true_labels = np.array(true_labels)
    preds = np.array(preds)
    for label in np.unique(true_labels):
        mask = true_labels == label
        correct = (preds[mask] == true_labels[mask]).sum()
        letter_acc[le_letters.inverse_transform([label])[0]] = correct / mask.sum()
    fold_letter_accuracies.append(letter_acc)
    history_all_folds.append(history)

# ====== Overall Results ======
overall_acc = np.mean(fold_accuracies)
print(f"\n=== Overall Accuracy from {FOLDS}-Fold CV: {overall_acc:.4f} ===")

# Average letter-wise accuracy
avg_letter_acc = {}
for letter in fold_letter_accuracies[0]:
    avg_letter_acc[letter] = np.mean([f[letter] for f in fold_letter_accuracies])

print("\nAverage Letter-wise Accuracies:")
for letter, acc in sorted(avg_letter_acc.items()):
    print(f"{letter}: {acc:.4f}")

# ====== Visualizations ======
for metric in ['loss', 'reward', 'epsilon']:
    plt.figure(figsize=(12, 8))
    for fold, hist in enumerate(history_all_folds):
        plt.plot(hist['epoch'], hist[metric], label=f'Fold {fold+1}')
    plt.title(f'{metric.title()} per Epoch Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel(metric.title())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Fold Accuracies
plt.figure(figsize=(10, 6))
plt.plot(range(1, FOLDS+1), fold_accuracies, 'bo-', label='Fold Accuracy')
plt.axhline(y=overall_acc, color='r', linestyle='--', label=f'Overall Accuracy: {overall_acc:.4f}')
plt.title('Accuracy per Fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Letter-wise Accuracy
letters = list(avg_letter_acc.keys())
accs = list(avg_letter_acc.values())

plt.figure(figsize=(12, 8))
bars = plt.bar(letters, accs, color='skyblue')
plt.axhline(y=overall_acc, color='r', linestyle='-', label=f'Overall Accuracy: {overall_acc:.4f}')
plt.title('Average Letter-wise Recognition Accuracy')
plt.xlabel('Letters')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1.1)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom', rotation=0)

plt.legend()
plt.tight_layout()
plt.show()
