import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

class HybridAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        
        # Initialize a model to predict when RF is likely correct/incorrect
        self.hidden_size = 128
        self.weights1 = np.random.randn(state_size, self.hidden_size) / np.sqrt(state_size)
        self.weights2 = np.random.randn(self.hidden_size, 1) / np.sqrt(self.hidden_size)
        
        # Track RF performance per class
        self.rf_class_performance = np.zeros(action_size) + 0.5  # Start with 0.5 trust for each class
        self.class_counts = np.zeros(action_size) + 1  # Avoid division by zero
        
        # Track examples
        self.example_buffer = []
        self.buffer_size = 1000

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def predict_rf_correctness(self, state):
        """Predict probability that RF is correct for this sample"""
        hidden = self.relu(np.dot(state, self.weights1))
        output = self.sigmoid(np.dot(hidden, self.weights2))
        return output[0]
    
    def act(self, state, rf_suggestion):
        """Decide whether to trust RF or try something else"""
        # Predict RF correctness for this state
        rf_trustworthiness = self.predict_rf_correctness(state)
        
        # Get RF class-specific performance
        rf_class_trust = self.rf_class_performance[rf_suggestion]
        
        # Combine the two factors
        trust_score = 0.7 * rf_trustworthiness + 0.3 * rf_class_trust
        
        # If we trust RF suggestion, use it
        if random.random() < trust_score:
            return rf_suggestion
        
        # Otherwise, try based on class performance (or random if all bad)
        class_probs = self.rf_class_performance.copy()
        # Reduce probability of selecting RF suggestion again
        class_probs[rf_suggestion] *= 0.5
        
        # Normalize to get probability distribution
        if np.sum(class_probs) > 0:
            class_probs = class_probs / np.sum(class_probs)
            return np.random.choice(self.action_size, p=class_probs)
        else:
            # If all classes have 0 probability, choose randomly
            return random.randint(0, self.action_size - 1)
    
    def learn(self, state, rf_suggestion, true_label):
        """Learn from the experience"""
        # Update RF class performance
        is_rf_correct = 1.0 if rf_suggestion == true_label else 0.0
        self.rf_class_performance[rf_suggestion] = (
            (self.rf_class_performance[rf_suggestion] * self.class_counts[rf_suggestion] + is_rf_correct) / 
            (self.class_counts[rf_suggestion] + 1)
        )
        self.class_counts[rf_suggestion] += 1
        
        # Store example for training the RF correctness predictor
        self.example_buffer.append((state, is_rf_correct))
        if len(self.example_buffer) > self.buffer_size:
            self.example_buffer.pop(0)
        
        # Only train the predictor every 50 examples to stabilize learning
        if len(self.example_buffer) % 50 == 0:
            self.train_rf_predictor()
    
    def train_rf_predictor(self):
        """Train the model to predict when RF is correct/incorrect"""
        if len(self.example_buffer) < 100:
            return  # Wait until we have enough examples
        
        # Create mini-batches
        batch_size = min(64, len(self.example_buffer))
        for _ in range(5):  # Do 5 mini-batch updates
            # Sample from buffer
            indices = np.random.choice(len(self.example_buffer), batch_size, replace=False)
            
            # Prepare batch
            states = np.array([self.example_buffer[i][0] for i in indices])
            targets = np.array([[self.example_buffer[i][1]] for i in indices])
            
            # Forward pass
            hidden = self.relu(np.dot(states, self.weights1))
            outputs = self.sigmoid(np.dot(hidden, self.weights2))
            
            # Compute error
            errors = targets - outputs
            
            # Backward pass
            d_outputs = errors
            d_hidden = np.dot(d_outputs, self.weights2.T)
            d_hidden[hidden <= 0] = 0  # ReLU derivative
            
            # Update weights with small learning rate for stability
            self.weights2 += self.lr * np.dot(hidden.T, d_outputs)
            self.weights1 += self.lr * np.dot(states.T, d_hidden)

# Load and prepare data
df = pd.read_csv("extended_big_output_cv2.csv")
df = df.drop(columns=["Image"])

# Encode labels
le_letters = LabelEncoder()
df["Letters_encoded"] = le_letters.fit_transform(df["Letters"])
df["RF_Output_encoded"] = le_letters.transform(df["RF_Output"])

# Features and labels
X = df.drop(columns=["Letters", "Letters_encoded", "RF_Output", "RF_Output_encoded"]).values.astype(np.float32)
y = df["Letters_encoded"].values.astype(int)
rf_outputs = df["RF_Output_encoded"].values.astype(int)

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test, rf_train, rf_test = train_test_split(
    X, y, rf_outputs, test_size=0.2, random_state=42, stratify=y
)

# Create agent
state_size = X.shape[1]
action_size = len(np.unique(y))
agent = HybridAgent(state_size=state_size, action_size=action_size, learning_rate=0.001)

# Training parameters
epochs = 20
train_accuracy_history = []
test_accuracy_history = []
rf_accuracy_history = []

print("Starting training...")

for epoch in range(epochs):
    # Shuffle training data
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    
    correct_train = 0
    
    # Training
    for i, idx in enumerate(indices):
        state = X_train[idx]
        true_label = y_train[idx]
        rf_suggestion = rf_train[idx]
        
        # Get action
        action = agent.act(state, rf_suggestion)
        
        # Check if correct
        if action == true_label:
            correct_train += 1
        
        # Learn from experience
        agent.learn(state, rf_suggestion, true_label)
    
    train_accuracy = correct_train / len(X_train)
    
    # Evaluate on test set
    correct_test = 0
    rf_correct = 0
    
    for i in range(len(X_test)):
        state = X_test[i]
        true_label = y_test[i]
        rf_suggestion = rf_test[i]
        
        action = agent.act(state, rf_suggestion)
        
        if action == true_label:
            correct_test += 1
        if rf_suggestion == true_label:
            rf_correct += 1
    
    test_accuracy = correct_test / len(X_test)
    rf_accuracy = rf_correct / len(X_test)
    
    train_accuracy_history.append(train_accuracy)
    test_accuracy_history.append(test_accuracy)
    rf_accuracy_history.append(rf_accuracy)
    
    print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f} | RF Acc: {rf_accuracy:.4f}")
    
    # Early stopping based on significant improvement over RF
    if test_accuracy > rf_accuracy * 1.1:
        print(f"Stopping early: Agent is performing 10% better than RF")
        break

# Final evaluation on all data
all_true_labels = []
agent_preds = []
rf_preds = []

for i in range(len(X)):
    state = X[i]
    true_label = y[i]
    rf_suggestion = rf_outputs[i]
    
    action = agent.act(state, rf_suggestion)
    
    all_true_labels.append(true_label)
    agent_preds.append(action)
    rf_preds.append(rf_suggestion)

# Calculate accuracy
agent_accuracy = accuracy_score(all_true_labels, agent_preds)
rf_accuracy = accuracy_score(all_true_labels, rf_preds)

print("\n--- Final Evaluation ---")
print(f"Agent Accuracy: {agent_accuracy:.4f}")
print(f"RandomForest Accuracy: {rf_accuracy:.4f}")
print(f"Difference: {(agent_accuracy - rf_accuracy) * 100:.2f}%")

# Per-class accuracy
letter_accuracies = {}
rf_letter_accuracies = {}

for class_idx in np.unique(y):
    class_indices = [i for i, label in enumerate(all_true_labels) if label == class_idx]
    
    if class_indices:
        # Agent accuracy
        agent_correct = sum(1 for i in class_indices if agent_preds[i] == all_true_labels[i])
        agent_acc = agent_correct / len(class_indices)
        
        # RF accuracy
        rf_correct = sum(1 for i in class_indices if rf_preds[i] == all_true_labels[i])
        rf_acc = rf_correct / len(class_indices)
        
        letter = le_letters.inverse_transform([class_idx])[0]
        letter_accuracies[letter] = agent_acc
        rf_letter_accuracies[letter] = rf_acc
        
        print(f"Letter '{letter}': Agent: {agent_acc:.4f}, RF: {rf_acc:.4f}, Diff: {(agent_acc - rf_acc) * 100:.1f}%")

# Save predictions
predictions_df = df.copy()
predictions_df["Agent_Prediction"] = le_letters.inverse_transform(agent_preds)
predictions_df.to_csv("hybrid_agent_predictions.csv", index=False)

# Plot results
plt.figure(figsize=(15, 10))

# Accuracy over time
plt.subplot(2, 2, 1)
plt.plot(range(1, len(train_accuracy_history) + 1), train_accuracy_history, 'b-o', label='Train')
plt.plot(range(1, len(test_accuracy_history) + 1), test_accuracy_history, 'g-o', label='Test')
plt.plot(range(1, len(rf_accuracy_history) + 1), rf_accuracy_history, 'r--', label='RF')
plt.title('Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Per-letter comparison
plt.subplot(2, 2, 2)
sorted_items = sorted(letter_accuracies.items())
letters = [item[0] for item in sorted_items]
agent_accs = [item[1] for item in sorted_items]
rf_accs = [rf_letter_accuracies[letter] for letter in letters]

x = np.arange(len(letters))
width = 0.35

plt.bar(x - width/2, agent_accs, width, label='Agent', color='skyblue')
plt.bar(x + width/2, rf_accs, width, label='RandomForest', color='lightgreen')

plt.ylim(0, 1)
plt.title("Per-Letter Accuracy Comparison")
plt.xlabel("Letters")
plt.ylabel("Accuracy")
plt.xticks(x, letters, rotation=90)
plt.legend()
plt.grid(axis='y')

# RF class performance
plt.subplot(2, 2, 3)
class_performance = [(le_letters.inverse_transform([i])[0], agent.rf_class_performance[i]) 
                      for i in range(len(agent.rf_class_performance))]
class_performance.sort(key=lambda x: x[1], reverse=True)

letters_sorted = [item[0] for item in class_performance]
performance_values = [item[1] for item in class_performance]

plt.bar(range(len(letters_sorted)), performance_values, color='lightcoral')
plt.xticks(range(len(letters_sorted)), letters_sorted, rotation=90)
plt.title("RF Performance by Letter (Agent's View)")
plt.xlabel("Letters")
plt.ylabel("Estimated Accuracy")
plt.grid(axis='y')

plt.tight_layout()
plt.savefig("hybrid_agent_results.png")
print("\nPlots saved.")