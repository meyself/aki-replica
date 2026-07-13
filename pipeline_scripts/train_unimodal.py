"""
train_unimodal.py — Four Unimodal Baselines for AKI Prediction

Key fixes vs previous version:
  1. Loads REAL data from output/train_cohort.csv and output/test_cohort.csv
  2. BioMedBERT: pre-tokenizes ALL chunks ONCE before training (fixes 17h runtime)
  3. BioMedBERT: uses safetensors format (fixes CVE-2025-32434 vulnerability)
  4. BioMedBERT: caps at 3 chunks per patient (prevents memory/time explosion)
  5. LSTM: weights_only=True on torch.load (fixes security warning)
  6. IS_MOCK_BERT: tries HuggingFace directly (not file-existence check)
"""

import os
import ast
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as tdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Global flags
# ---------------------------------------------------------------------------
USE_FINETUNING         = True
MAX_CHUNKS_PER_PATIENT = 3   # cap note chunks per patient for BERT training

os.makedirs('../models',     exist_ok=True)
os.makedirs('../embeddings', exist_ok=True)

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ---------------------------------------------------------------------------
# 1. Load real cohort data
# ---------------------------------------------------------------------------
print("\nLoading real cohort data...")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
TRAIN_PATH = os.path.join(OUTPUT_DIR, 'train_cohort.csv')
TEST_PATH  = os.path.join(OUTPUT_DIR, 'test_cohort.csv')

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"train_cohort.csv not found at {TRAIN_PATH}\n"
        "Run run_pipeline.py first."
    )

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"  Train rows: {len(train_df):,}")
print(f"  Test  rows: {len(test_df):,}")

def parse_ts_matrix(s):
    try:
        return np.array(ast.literal_eval(s), dtype=np.float32)
    except Exception:
        return np.zeros((6, 72), dtype=np.float32)

def parse_notes(s):
    if pd.isna(s) or s == '[]':
        return []
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return []

print("  Parsing time-series matrices...")
train_df['ts_parsed'] = train_df['ts_matrix'].apply(parse_ts_matrix)
test_df['ts_parsed']  = test_df['ts_matrix'].apply(parse_ts_matrix)

print("  Parsing note tokens...")
train_df['notes_parsed'] = train_df['note_tokens'].apply(parse_notes)
test_df['notes_parsed']  = test_df['note_tokens'].apply(parse_notes)

X_train_raw = np.stack(train_df['ts_parsed'].values)
X_test_raw  = np.stack(test_df['ts_parsed'].values)
y_train     = train_df['aki_target'].values.astype(int)
y_test      = test_df['aki_target'].values.astype(int)

train_subjects = train_df['subject_id'].tolist()
test_subjects  = test_df['subject_id'].tolist()
train_notes    = train_df['notes_parsed'].tolist()
test_notes     = test_df['notes_parsed'].tolist()

print(f"\nTrain shapes — TS Raw: {X_train_raw.shape}, Labels: {y_train.shape}")
print(f"Test  shapes — TS Raw: {X_test_raw.shape},  Labels: {y_test.shape}")
print(f"Class dist (train) — Positives: {np.sum(y_train):,}  "
      f"Negatives: {len(y_train)-np.sum(y_train):,}  "
      f"({np.mean(y_train):.1%} positive)")
print(f"Class dist (test)  — Positives: {np.sum(y_test):,}  "
      f"Negatives: {len(y_test)-np.sum(y_test):,}  "
      f"({np.mean(y_test):.1%} positive)")

# ---------------------------------------------------------------------------
# 2. Feature engineering for LR / XGBoost
# ---------------------------------------------------------------------------
def extract_engineered(raw_tensor):
    result = []
    for matrix in raw_tensor:
        flattened   = matrix.flatten()
        values      = matrix[:, :36]
        masks       = matrix[:, 36:]
        min_v       = np.min(values,  axis=0)
        max_v       = np.max(values,  axis=0)
        mean_v      = np.mean(values, axis=0)
        std_v       = np.std(values,  axis=0)
        n           = values.shape[0]
        m3          = np.sum((values - mean_v)**3, axis=0) / n
        m2          = np.sum((values - mean_v)**2, axis=0) / n
        skew_v      = np.nan_to_num(m3 / (m2**1.5 + 1e-8))
        count_obs   = np.sum(masks == 0, axis=0)
        engineered  = np.concatenate([min_v, max_v, mean_v, std_v,
                                      skew_v, count_obs])
        result.append(np.concatenate([flattened, engineered]))
    return np.array(result, dtype=np.float32)

print("\nBuilding engineered features for LR/XGBoost...")
X_train_eng = extract_engineered(X_train_raw)
X_test_eng  = extract_engineered(X_test_raw)
print(f"  Engineered shape: {X_train_eng.shape}")

# ---------------------------------------------------------------------------
# Results collector
# ---------------------------------------------------------------------------
results = []

def evaluate_model(name, y_true, y_pred, y_prob):
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred,  zero_division=0)
    rec   = recall_score(y_true, y_pred,     zero_division=0)
    try:
        auc   = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auc, auprc = 0.0, 0.0
    cm = confusion_matrix(y_true, y_pred)
    results.append({'Model': name, 'Accuracy': round(acc,4),
                    'Precision': round(prec,4), 'Recall': round(rec,4),
                    'AUROC': round(auc,4), 'AUPRC': round(auprc,4)})
    print(f"\n[{name}]")
    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  AUROC={auc:.4f}  AUPRC={auprc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

# ---------------------------------------------------------------------------
# Model 1: Logistic Regression
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Model 1: Logistic Regression")
print("="*60)

lr_model = LogisticRegression(solver='lbfgs', max_iter=1000,
                               class_weight=None)
lr_model.fit(X_train_eng, y_train)
y_pred_lr = lr_model.predict(X_test_eng)
y_prob_lr = lr_model.predict_proba(X_test_eng)[:, 1]
evaluate_model('Logistic Regression', y_test, y_pred_lr, y_prob_lr)

with open('../models/lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("  Saved: models/lr_model.pkl")

# ---------------------------------------------------------------------------
# Model 2: XGBoost
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Model 2: XGBoost")
print("="*60)

n_pos = int(np.sum(y_train))
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
print(f"  scale_pos_weight = {scale_pos_weight:.3f}  "
      f"({n_neg} neg / {n_pos} pos)")

xgb_model = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, eval_metric='auc',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbosity=0
)
xgb_model.fit(X_train_eng, y_train)
y_pred_xgb = xgb_model.predict(X_test_eng)
y_prob_xgb = xgb_model.predict_proba(X_test_eng)[:, 1]
evaluate_model('XGBoost', y_test, y_pred_xgb, y_prob_xgb)

with open('../models/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("  Saved: models/xgb_model.pkl")

# ---------------------------------------------------------------------------
# Model 3: LSTM
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Model 3: LSTM")
print("="*60)

class TS_LSTM(nn.Module):
    def __init__(self, input_size=72, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=input_size, hidden_size=hidden_dim,
                               batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        hidden  = hn[-1]
        dropped = self.dropout(hidden)
        logits  = self.fc(dropped)
        return logits, hidden

lstm_model = TS_LSTM().to(device)
optimizer  = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
criterion  = nn.CrossEntropyLoss()

class TSDataset(tdata.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

N_TRAIN           = len(train_subjects)
USE_EARLY_STOP    = N_TRAIN >= 10

if USE_EARLY_STOP:
    val_n    = max(1, int(0.1 * N_TRAIN))
    train_n  = N_TRAIN - val_n
    X_tr, y_tr   = X_train_raw[:train_n], y_train[:train_n]
    X_val, y_val = X_train_raw[train_n:], y_train[train_n:]
    val_loader   = tdata.DataLoader(TSDataset(X_val, y_val),
                                    batch_size=256, shuffle=False)
    print(f"  Early stopping — train: {train_n:,}  val: {val_n:,}  patience: 10")
else:
    X_tr, y_tr = X_train_raw, y_train
    print("  Skipping early stopping (small dataset)")

train_loader = tdata.DataLoader(TSDataset(X_tr, y_tr),
                                batch_size=128, shuffle=True)

best_val_loss  = float('inf')
patience_count = 0
PATIENCE       = 10
EPOCHS         = 50

for epoch in range(EPOCHS):
    lstm_model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, _ = lstm_model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    if USE_EARLY_STOP:
        lstm_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = lstm_model(xb)
                val_loss += criterion(logits, yb).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:03d}/{EPOCHS}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(lstm_model.state_dict(), '../models/lstm_model.pt')
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    else:
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:03d}/{EPOCHS}  loss={train_loss:.4f}")
        torch.save(lstm_model.state_dict(), '../models/lstm_model.pt')

# Reload best checkpoint — weights_only=True fixes security warning
lstm_model.load_state_dict(
    torch.load('../models/lstm_model.pt',
               map_location=device, weights_only=True)
)
lstm_model.eval()

all_probs, all_preds, all_hiddens_test = [], [], []
test_loader = tdata.DataLoader(TSDataset(X_test_raw, y_test),
                               batch_size=256, shuffle=False)
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        logits, hidden = lstm_model(xb)
        all_probs.extend(torch.softmax(logits, dim=1)[:,1].cpu().numpy())
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_hiddens_test.append(hidden.cpu().numpy())

y_prob_lstm  = np.array(all_probs)
y_pred_lstm  = np.array(all_preds)
test_hiddens = np.vstack(all_hiddens_test)

evaluate_model('LSTM', y_test, y_pred_lstm, y_prob_lstm)

all_hiddens_train = []
train_loader_seq  = tdata.DataLoader(TSDataset(X_train_raw, y_train),
                                     batch_size=256, shuffle=False)
lstm_model.eval()
with torch.no_grad():
    for xb, _ in train_loader_seq:
        xb = xb.to(device)
        _, hidden = lstm_model(xb)
        all_hiddens_train.append(hidden.cpu().numpy())
train_hiddens = np.vstack(all_hiddens_train)

lstm_embeddings = {}
for idx, subj in enumerate(train_subjects):
    lstm_embeddings[subj] = train_hiddens[idx]
for idx, subj in enumerate(test_subjects):
    lstm_embeddings[subj] = test_hiddens[idx]

with open('../embeddings/lstm_hidden_states.pkl', 'wb') as f:
    pickle.dump(lstm_embeddings, f)
print(f"  Saved: models/lstm_model.pt")
print(f"  Saved: embeddings/lstm_hidden_states.pkl  "
      f"({len(lstm_embeddings)} patients, shape (64,) each)")

# ---------------------------------------------------------------------------
# Model 4: BioMedBERT
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Model 4: BioMedBERT")
print("="*60)

BERT_MODEL_NAME = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
_is_mock_bert   = True
bert_tokenizer  = None
bert_backbone   = None

if USE_FINETUNING:
    print("  Attempting to load BioMedBERT from HuggingFace...")
    try:
        from transformers import AutoTokenizer, AutoModel
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        # use_safetensors=True avoids CVE-2025-32434 torch.load vulnerability
        bert_backbone  = AutoModel.from_pretrained(BERT_MODEL_NAME,
                                                   use_safetensors=True)
        _is_mock_bert  = False
        print("  BioMedBERT loaded successfully — entering FINE-TUNING mode")
    except Exception as e:
        print(f"  BioMedBERT load failed: {e}")
        print("  Falling back to frozen/mock mode")
        _is_mock_bert = True
else:
    print("  USE_FINETUNING=False — skipping to mock mode")

# ---------------------------------------------------------------------------
# Branch A: Fine-tune BioMedBERT
# ---------------------------------------------------------------------------
if not _is_mock_bert:

    class BioMedBERTClassifier(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone   = backbone
            self.classifier = nn.Linear(768, 2)

        def forward(self, input_ids, attention_mask):
            outputs = self.backbone(input_ids=input_ids,
                                    attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            logits  = self.classifier(cls_emb)
            return logits, cls_emb

    bert_clf = BioMedBERTClassifier(bert_backbone).to(device)
    print(f"  BioMedBERT on device: {next(bert_clf.parameters()).device}")

    bert_optimizer = torch.optim.AdamW(bert_clf.parameters(), lr=2e-5)
# Weight positive class by imbalance ratio to prevent all-negative predictions
pos_weight = torch.tensor([1.0, n_neg/n_pos]).to(device)
bert_criterion = nn.CrossEntropyLoss(weight=pos_weight)

    # Build training pairs — cap at MAX_CHUNKS_PER_PATIENT per patient
    bert_train_pairs = []
    for subj_idx, chunks in enumerate(train_notes):
        label = int(y_train[subj_idx])
        for chunk in chunks[:MAX_CHUNKS_PER_PATIENT]:
            if isinstance(chunk, list):
                chunk = ' '.join(str(t) for t in chunk)
            if isinstance(chunk, str) and len(chunk.strip()) > 0:
                bert_train_pairs.append((chunk, label))

    print(f"  Training pairs (chunks × patients): {len(bert_train_pairs):,}")

    # Pre-tokenize dataset class — tokenization happens ONCE before training
    class NoteDataset(tdata.Dataset):
        def __init__(self, pairs, tokenizer, max_len=512):
            print(f"    Pre-tokenizing {len(pairs):,} chunks (one-time)...")
            texts  = [p[0] for p in pairs]
            labels = [p[1] for p in pairs]

            BATCH = 512
            all_ids, all_masks = [], []
            for i in range(0, len(texts), BATCH):
                enc = tokenizer(
                    texts[i:i+BATCH],
                    max_length=max_len,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                all_ids.append(enc['input_ids'])
                all_masks.append(enc['attention_mask'])
                if (i // BATCH + 1) % 20 == 0:
                    print(f"      {min(i+BATCH, len(texts)):,} / "
                          f"{len(texts):,} tokenized...")

            self.input_ids      = torch.cat(all_ids,   dim=0)
            self.attention_mask = torch.cat(all_masks, dim=0)
            self.labels         = torch.tensor(labels, dtype=torch.long)
            print(f"    Done — input_ids shape: {self.input_ids.shape}")

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return (self.input_ids[idx],
                    self.attention_mask[idx],
                    self.labels[idx])

    BERT_BATCH  = 32 if torch.cuda.is_available() else 8
    note_loader = tdata.DataLoader(
        NoteDataset(bert_train_pairs, bert_tokenizer),
        batch_size=BERT_BATCH,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # Fine-tuning loop — GPU only, no CPU tokenization per batch
    bert_clf.train()
    for epoch in range(3):
        epoch_loss = 0.0
        n_batches  = 0
        for input_ids, attention_mask, labels in note_loader:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels         = labels.to(device)
            bert_optimizer.zero_grad()
            logits, _ = bert_clf(input_ids, attention_mask)
            loss = bert_criterion(logits, labels)
            loss.backward()
            bert_optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        print(f"  Fine-tune epoch {epoch+1}/3  "
              f"loss={epoch_loss/max(n_batches,1):.4f}")

    # Extract embeddings for all patients
    def extract_bert_patient(note_chunks):
        bert_clf.eval()
        all_probs, all_cls = [], []
        for chunk in note_chunks[:MAX_CHUNKS_PER_PATIENT]:
            if isinstance(chunk, list):
                chunk = ' '.join(str(t) for t in chunk)
            if not isinstance(chunk, str) or len(chunk.strip()) == 0:
                continue
            enc = bert_tokenizer(
                chunk, max_length=512, truncation=True,
                padding='max_length', return_tensors='pt'
            )
            with torch.no_grad():
                logits, cls_emb = bert_clf(
                    enc['input_ids'].to(device),
                    enc['attention_mask'].to(device)
                )
            all_probs.append(torch.softmax(logits, dim=1)[0,1].item())
            all_cls.append(cls_emb.squeeze(0).cpu().numpy())

        if not all_probs:
            return 0.5, np.zeros(768, dtype=np.float32)
        n, c    = len(all_probs), 2
        P_final = (max(all_probs) + (n/c)*np.mean(all_probs)) / (1 + n/c)
        return P_final, np.mean(all_cls, axis=0)

    bert_embeddings_dict = {}
    all_subjects       = list(train_subjects) + list(test_subjects)
    all_notes_combined = list(train_notes)    + list(test_notes)

    print(f"  Extracting embeddings for {len(all_subjects):,} patients...")
    for i, (subj_id, chunks) in enumerate(zip(all_subjects, all_notes_combined)):
        P_final, cls_emb = extract_bert_patient(chunks)
        bert_embeddings_dict[subj_id] = {
            'cls_embedding': cls_emb,
            'P_final':       P_final
        }
        if (i + 1) % 5000 == 0:
            print(f"    {i+1:,} / {len(all_subjects):,} patients done...")

    y_prob_bert, y_pred_bert = [], []
    for subj_id in test_subjects:
        p = bert_embeddings_dict[subj_id]['P_final']
        y_prob_bert.append(p)
        y_pred_bert.append(1 if p > 0.5 else 0)

    evaluate_model('BioMedBERT', y_test, y_pred_bert, y_prob_bert)

    # Save — load via BioMedBERTClassifier.load_state_dict(), not from_pretrained
    torch.save(bert_clf.state_dict(), '../models/biomedbert_finetuned.pt')
    print("  Saved: models/biomedbert_finetuned.pt")

    bert_embeddings = {'embeddings': bert_embeddings_dict, 'IS_MOCK_BERT': False}
    print("  IS_MOCK_BERT: False")

# ---------------------------------------------------------------------------
# Branch B: Mock mode
# ---------------------------------------------------------------------------
else:
    print("  BioMedBERT running in FROZEN/MOCK mode.")
    bert_embeddings_dict = {}
    y_prob_bert, y_pred_bert = [], []

    for subj, chunks in zip(list(train_subjects) + list(test_subjects),
                             list(train_notes)    + list(test_notes)):
        n       = max(len(chunks), 1)
        probs   = np.random.uniform(0.1, 0.9, size=n)
        P_final = (max(probs) + (n/2)*np.mean(probs)) / (1 + n/2)
        bert_embeddings_dict[subj] = {
            'cls_embedding': np.random.randn(768).astype(np.float32),
            'P_final':       float(P_final)
        }

    for subj in test_subjects:
        p = bert_embeddings_dict[subj]['P_final']
        y_prob_bert.append(p)
        y_pred_bert.append(1 if p > 0.5 else 0)

    evaluate_model('BioMedBERT (Mock)', y_test, y_pred_bert, y_prob_bert)

    bert_embeddings = {'embeddings': bert_embeddings_dict, 'IS_MOCK_BERT': True}
    print("\n  [WARNING] IS_MOCK_BERT=True. Regenerate with real BioMedBERT "
          "before running train_fusion.py")

with open('../embeddings/biomedbert_cls_embeddings.pkl', 'wb') as f:
    pickle.dump(bert_embeddings, f)
print(f"  Saved: embeddings/biomedbert_cls_embeddings.pkl  "
      f"(IS_MOCK_BERT={bert_embeddings['IS_MOCK_BERT']})")

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("FINAL RESULTS TABLE")
print("="*60)

results_df = pd.DataFrame(results)
paper_auroc = {
    'Logistic Regression': 0.832,
    'XGBoost':             0.855,
    'LSTM':                0.873,
    'BioMedBERT':          0.742,
    'BioMedBERT (Mock)':   'N/A'
}
results_df['Paper_AUROC_Ref'] = results_df['Model'].map(paper_auroc)
print(results_df.to_string(index=False))
print("\nPaper's expected ranking: LSTM > XGBoost > LR > BioMedBERT (AUROC)")
print("Note: AUPRC is primary metric due to class imbalance")

results_df.to_csv('../output/unimodal_results.csv', index=False)
print("\nSaved: output/unimodal_results.csv")

# ---------------------------------------------------------------------------
# Embedding verification
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("EMBEDDING VERIFICATION")
print("="*60)

with open('../embeddings/lstm_hidden_states.pkl', 'rb') as f:
    lstm_emb = pickle.load(f)
sample_lstm = list(lstm_emb.values())[0]
print(f"LSTM embeddings: {len(lstm_emb)} patients, "
      f"shape per patient: {sample_lstm.shape}")

with open('../embeddings/biomedbert_cls_embeddings.pkl', 'rb') as f:
    bert_emb = pickle.load(f)
sample_bert = list(bert_emb['embeddings'].values())[0]
print(f"BERT embeddings: {len(bert_emb['embeddings'])} patients, "
      f"shape per patient: {sample_bert['cls_embedding'].shape}")
print(f"IS_MOCK_BERT: {bert_emb['IS_MOCK_BERT']}")

print("\ntrain_unimodal.py complete.")