import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim, re
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss
from collections import Counter

#peaked at 0.818 EPOCH-5
# Config
TRAIN_PATH = r".\data\train-data.csv"
TEST_PATH = r".\data\test-data.csv"
TARGET_COLS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']
N_LABELS = len(TARGET_COLS)
MAX_VOCAB = 25000
SEQ_LEN = 300
BATCH_SIZE = 64
EMBED_DIM = 256
HIDDEN_DIM = 128
LR = 0.001
EPOCHS = 10
THRESHOLD = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Utils
def tokenize(text):
    if not isinstance(text, str): return []
    return re.findall(r'\w+', text.lower())

def vectorize(texts, vocab, max_len):
    indices = [[vocab.get(w, 1) for w in tokenize(t)][:max_len] for t in texts]
    return np.array([x + [0]*(max_len-len(x)) for x in indices])

# Load
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
y_train = train_df[TARGET_COLS].values.astype(np.float32)
y_test = test_df[TARGET_COLS].values.astype(np.float32)

# Vocab
vocab_cnt = Counter(w for txt in train_df['abstractText'] for w in tokenize(txt))
vocab = {w: i+2 for i, (w, _) in enumerate(vocab_cnt.most_common(MAX_VOCAB - 2))}
vocab['<PAD>'] = 0; vocab['<UNK>'] = 1

# Vectorize
X_train = vectorize(train_df['abstractText'], vocab, SEQ_LEN)
X_test = vectorize(test_df['abstractText'], vocab, SEQ_LEN)

# Dataloaders
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32))
test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Model
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(HIDDEN_DIM * 2, N_LABELS)
        
    def forward(self, x):
        embedded = self.emb(x)
        _, (h, _) = self.lstm(embedded)
        hidden = torch.cat((h[-2], h[-1]), dim=1) 
        return self.fc(hidden)

model = RNN().to(DEVICE)
opt = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()

# History
history = {
    'loss': [], 'micro_f1': [], 'macro_f1': [], 'weighted_f1': [],
    'exact_match': [], 'hamming': []
}

# Train
print("\nStarting Training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
        opt.zero_grad()
        preds = model(x_b)
        loss = loss_fn(preds, y_b)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    
    # Evaluate
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            logits = model(x_b)
            probs = torch.sigmoid(logits)
            all_p.append((probs > THRESHOLD).float().cpu().numpy())
            all_t.append(y_b.cpu().numpy())
            
    p_cat = np.vstack(all_p)
    t_cat = np.vstack(all_t)
    
    # Calculate
    mic_f1 = f1_score(t_cat, p_cat, average='micro')
    mac_f1 = f1_score(t_cat, p_cat, average='macro')
    wgt_f1 = f1_score(t_cat, p_cat, average='weighted')
    emr = accuracy_score(t_cat, p_cat)
    ham = hamming_loss(t_cat, p_cat)
    
    # Store
    history['loss'].append(avg_loss)
    history['micro_f1'].append(mic_f1)
    history['macro_f1'].append(mac_f1)
    history['weighted_f1'].append(wgt_f1)
    history['exact_match'].append(emr)
    history['hamming'].append(ham)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Micro F1: {mic_f1:.4f}")

# Plotting 
epochs_range = range(1, EPOCHS + 1)

#Training Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history['loss'], label='Train Loss', marker='o', color='blue')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# F1 Scores
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history['micro_f1'], label='Micro F1', marker='o')
plt.plot(epochs_range, history['macro_f1'], label='Macro F1', marker='s')
plt.plot(epochs_range, history['weighted_f1'], label='Weighted F1', marker='^')
plt.title('F1 Scores Comparison')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.grid(True)
plt.legend()
plt.show()

# Global Metrics
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history['exact_match'], label='Exact Match Ratio', color='green', marker='o')
plt.plot(epochs_range, history['hamming'], label='Hamming Loss', color='red', marker='x')
plt.title('Global Metrics (Exact Match & Hamming)')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.grid(True)
plt.legend()
plt.show()