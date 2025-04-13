import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# ========== Basitleştirilmiş QLearning Agent ================
class QLearningAgent:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.9, epsilon=0.5, epsilon_decay=0.95, min_epsilon=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # İki katmanlı basit bir yapı (lineer değil)
        self.hidden_size = 64
        self.weights1 = np.random.randn(state_size, self.hidden_size) * 0.01
        self.weights2 = np.random.randn(self.hidden_size, action_size) * 0.01
        
        # RF önerilerinin doğruluk geçmişi
        self.rf_correct_count = 0
        self.rf_total_count = 0
        self.rf_reliability = 0.5  # Başlangıçta %50 güven

    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, state):
        # İleri besleme hesaplaması
        hidden = self.relu(np.dot(state, self.weights1))
        q_values = np.dot(hidden, self.weights2)
        return q_values

    def act(self, state, rf_suggestion=None):
        # Epsilon-greedy politikası
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        q_values = self.forward(state)
        
        # RF önerisini kullan, ancak güvenilirliğine göre
        if rf_suggestion is not None and np.random.rand() < self.rf_reliability:
            # RF önerisini Q değerlerine bir boost olarak ekle
            boost_value = np.max(q_values) * 0.3  # Maksimum Q değerinin %30'u kadar boost
            q_values[rf_suggestion] += boost_value
        
        return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, true_label, rf_suggestion):
        # RF önerisinin doğruluğunu takip et
        if rf_suggestion is not None:
            self.rf_total_count += 1
            if rf_suggestion == true_label:
                self.rf_correct_count += 1
            
            # RF güvenilirliğini güncelle
            if self.rf_total_count > 10:  # En az 10 örnek gördükten sonra
                self.rf_reliability = self.rf_correct_count / self.rf_total_count
        
        # Q-Learning güncellemesi
        current_q_values = self.forward(state)
        next_q_values = self.forward(next_state)
        
        # Hedef Q değeri
        target = reward + self.gamma * np.max(next_q_values)
        
        # Hata (TD hatası)
        error = target - current_q_values[action]
        
        # Geri yayılım
        # 1. Çıkış katmanı gradyanı
        d_output = np.zeros_like(current_q_values)
        d_output[action] = error
        
        # 2. Gizli katman çıktısı
        hidden = self.relu(np.dot(state, self.weights1))
        
        # 3. Ağırlık güncellemeleri
        # Çıkış katmanı
        self.weights2 += self.lr * np.outer(hidden, d_output)
        
        # Gizli katman
        d_hidden = np.dot(d_output, self.weights2.T)
        d_hidden[hidden <= 0] = 0  # ReLU türevi
        self.weights1 += self.lr * np.outer(state, d_hidden)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# ========== Veri Hazırlığı ================
df = pd.read_csv("extended_big_output_cv2.csv")
df = df.drop(columns=["Image"])

# Orijinal harfleri sakla
original_letters = df["Letters"].values
original_rf_outputs = df["RF_Output"].values

# Etiketleri kodla
le_letters = LabelEncoder()
df["Letters_encoded"] = le_letters.fit_transform(df["Letters"])
df["RF_Output_encoded"] = le_letters.transform(df["RF_Output"])

# Özellikler ve etiketler
X = df.drop(columns=["Letters", "Letters_encoded", "RF_Output", "RF_Output_encoded"]).values.astype(np.float32)
y = df["Letters_encoded"].values.astype(int)
rf_outputs = df["RF_Output_encoded"].values.astype(int)

# Veriyi standartlaştır
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Eğitim/test ayrımı
X_train, X_test, y_train, y_test, rf_train, rf_test = train_test_split(
    X, y, rf_outputs, test_size=0.2, random_state=42, stratify=y
)

# ========== Ajan ve Eğitim ================
state_size = X.shape[1]
action_size = len(np.unique(y))
agent = QLearningAgent(
    state_size=state_size, 
    action_size=action_size,
    lr=0.01,
    gamma=0.9,
    epsilon=0.5,
    epsilon_decay=0.75,
    min_epsilon=0.1
)

# İyileştirilmiş ödül fonksiyonu
def calculate_reward(action, true_label, rf_suggestion):
    if action == true_label:
        return 10.0  # Doğru tahmin
    elif action == rf_suggestion and rf_suggestion != true_label:
        return -2.0  # RF yanlış ve biz onu takip etmişiz
    elif action != rf_suggestion and rf_suggestion == true_label:
        return -3.0  # RF doğru ama biz farklı tahmin etmişiz
    else:
        return -1.0  # Herkes yanlış

# Eğitim parametreleri
epochs = 15
reward_progress = []
test_accuracy_progress = []

print("Eğitim başlatılıyor...")

for epoch in range(epochs):
    total_reward = 0
    
    # Eğitim verilerini karıştır
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    
    for i, idx in enumerate(indices):
        state = X_train[idx]
        true_label = y_train[idx]
        rf_suggestion = rf_train[idx]
        
        # Eylemi seç
        action = agent.act(state, rf_suggestion=rf_suggestion)
        
        # Ödülü hesapla
        reward = calculate_reward(action, true_label, rf_suggestion)
        total_reward += reward
        
        # Sonraki durum (döngüsel olarak)
        next_idx = indices[(i + 1) % len(indices)]
        next_state = X_train[next_idx]
        
        # Öğrenme
        agent.learn(state, action, reward, next_state, true_label, rf_suggestion)
        
        # Epsilon azalt (her 100 adımda bir)
        if i % 100 == 0:
            agent.decay_epsilon()
    
    # Epoch sonu değerlendirmesi
    correct = 0
    for i in range(len(X_test)):
        state = X_test[i]
        true_label = y_test[i]
        rf_suggestion = rf_test[i]
        action = agent.act(state, rf_suggestion=rf_suggestion)
        if action == true_label:
            correct += 1
    
    test_accuracy = correct / len(X_test)
    test_accuracy_progress.append(test_accuracy)
    
    # RF güvenilirliği
    rf_reliability = agent.rf_reliability
    
    reward_progress.append(total_reward)
    print(f"Epoch {epoch+1}/{epochs} | Toplam ödül: {total_reward:.1f} | Test Doğruluğu: {test_accuracy:.4f} | RF Güvenilirliği: {rf_reliability:.4f}")

# ========== Doğruluk Hesaplama ================
print("\n--- Doğruluk Değerlendirmesi (Tüm Veri) ---")

# Tüm veri üzerinde değerlendirme
all_true_labels = []
agent_preds = []
rf_preds = []

for i in range(len(X)):
    state = X[i]
    true_label = y[i]
    rf_suggestion = rf_outputs[i]
    
    # Epsilon olmadan (değerlendirme modu)
    old_epsilon = agent.epsilon
    agent.epsilon = 0
    action = agent.act(state, rf_suggestion=rf_suggestion)
    agent.epsilon = old_epsilon
    
    all_true_labels.append(true_label)
    agent_preds.append(action)
    rf_preds.append(rf_suggestion)

# Doğruluk hesapla
agent_accuracy = accuracy_score(all_true_labels, agent_preds)
rf_accuracy = accuracy_score(all_true_labels, rf_preds)

print(f"Agent Doğruluğu: {agent_accuracy:.4f}")
print(f"RandomForest Doğruluğu: {rf_accuracy:.4f}")
print(f"Fark: {(agent_accuracy - rf_accuracy) * 100:.2f}%")

# Sınıf başına doğruluk
letter_accuracies = {}
rf_letter_accuracies = {}

for class_idx in np.unique(y):
    class_indices = [i for i, label in enumerate(all_true_labels) if label == class_idx]
    
    if class_indices:
        # Agent doğruluğu
        agent_correct = sum(1 for i in class_indices if agent_preds[i] == all_true_labels[i])
        agent_acc = agent_correct / len(class_indices)
        
        # RF doğruluğu
        rf_correct = sum(1 for i in class_indices if rf_preds[i] == all_true_labels[i])
        rf_acc = rf_correct / len(class_indices)
        
        letter = le_letters.inverse_transform([class_idx])[0]
        letter_accuracies[letter] = agent_acc
        rf_letter_accuracies[letter] = rf_acc
        
        print(f"Harf '{letter}': Agent: {agent_acc:.4f}, RF: {rf_acc:.4f}, Fark: {(agent_acc - rf_acc) * 100:.1f}%")

# CSV'ye tahminleri ekle
predictions_df = df.copy()
predictions_df["Agent_Prediction"] = le_letters.inverse_transform(agent_preds)

# Sonuçları kaydet
predictions_df.to_csv("agent_predictions2.csv", index=False)
print("\nPredictions saved to 'agent_predictions2.csv'")

# ========== Grafikleri Çiz ================
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), reward_progress, 'b-o')
plt.title('Eğitim Süreci - Toplam Ödül')
plt.xlabel('Epoch')
plt.ylabel('Toplam Ödül')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), test_accuracy_progress, 'g-o')
plt.title('Eğitim Süreci - Test Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.grid(True)
plt.tight_layout()
plt.savefig("training_progress2.png")

# Harf bazlı karşılaştırma grafiği
plt.figure(figsize=(14, 7))
sorted_items = sorted(letter_accuracies.items())
letters = [item[0] for item in sorted_items]
agent_accs = [item[1] for item in sorted_items]
rf_accs = [rf_letter_accuracies[letter] for letter in letters]

x = np.arange(len(letters))
width = 0.35

plt.bar(x - width/2, agent_accs, width, label='Agent', color='skyblue')
plt.bar(x + width/2, rf_accs, width, label='RandomForest', color='lightgreen')

plt.ylim(0, 1)
plt.title("Harf Bazlı Doğruluk Karşılaştırması")
plt.xlabel("Harfler")
plt.ylabel("Doğruluk")
plt.xticks(x, letters)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("letter_accuracy_comparison2.png")

# Karışıklık matrisi
cm = confusion_matrix(all_true_labels, agent_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", 
           xticklabels=le_letters.classes_, yticklabels=le_letters.classes_)
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Q-Learning Ajan Karışıklık Matrisi")
plt.tight_layout()
plt.savefig("confusion_matrix2.png")

print("\nGrafikler kaydedildi.")