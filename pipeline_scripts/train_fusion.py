"""
train_fusion.py — Multimodal Fusion Model for AKI Prediction

Optimized for speed:
  - Fusion model is tiny (832→256→64→2, ~220K params) — trains in minutes
  - All embeddings pre-loaded to GPU as tensors before training
  - batch_size=512 (safe for any GPU, fusion model uses far less VRAM than BERT)
  - Early stopping with patience=10 on validation loss
  - SHAP runs on small random sample (500 patients) for speed

Requires:
  - embeddings/lstm_hidden_states.pkl       (IS_MOCK_BERT irrelevant)
  - embeddings/biomedbert_cls_embeddings.pkl (must have IS_MOCK_BERT=False)
  - output/train_cohort.csv
  - output/test_cohort.csv
  - models/lr_model.pkl, xgb_model.pkl, lstm_model.pt (for comparison table)
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as tdata
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

os.makedirs('../models',     exist_ok=True)
os.makedirs('../embeddings', exist_ok=True)
os.makedirs('../output',     exist_ok=True)

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ── IS_MOCK_BERT guard ────────────────────────────────────────────────────────
print("\nChecking embedding files...")
with open('../embeddings/biomedbert_cls_embeddings.pkl', 'rb') as f:
    bert_data = pickle.load(f)

if bert_data['IS_MOCK_BERT']:
    print("WARNING: IS_MOCK_BERT=True — fusion will run but results are")
    print("         not meaningful for the notes modality.")
    print("         Fix BioMedBERT class weights and rerun train_unimodal.py")
    print("         then rerun this script for real fusion results.")
    print("         Proceeding anyway for structural verification...")
else:
    print("  IS_MOCK_BERT: False — real BioMedBERT embeddings confirmed ✓")

# ── Load cohort labels ────────────────────────────────────────────────────────
print("\nLoading cohort data...")
train_df = pd.read_csv('../output/train_cohort.csv',
                       usecols=['subject_id', 'aki_target'])
test_df  = pd.read_csv('../output/test_cohort.csv',
                       usecols=['subject_id', 'aki_target'])

train_subjects = train_df['subject_id'].tolist()
test_subjects  = test_df['subject_id'].tolist()
y_train        = train_df['aki_target'].values.astype(int)
y_test         = test_df['aki_target'].values.astype(int)

print(f"  Train: {len(train_subjects):,} patients")
print(f"  Test:  {len(test_subjects):,} patients")

# ── Load embeddings ───────────────────────────────────────────────────────────
print("\nLoading LSTM embeddings...")
with open('../embeddings/lstm_hidden_states.pkl', 'rb') as f:
    lstm_data = pickle.load(f)

print("Loading BioMedBERT embeddings...")
bert_embs = bert_data['embeddings']

# ── Build fusion embedding matrices ──────────────────────────────────────────
print("\nBuilding fusion embedding matrices...")

def get_fusion_embedding(subj_id):
    """Concatenate LSTM (64) + BERT CLS (768) = 832 dims."""
    lstm_emb = lstm_data.get(subj_id)
    bert_entry = bert_embs.get(subj_id)

    if lstm_emb is None:
        lstm_emb = np.zeros(64, dtype=np.float32)
    if bert_entry is None:
        bert_emb = np.zeros(768, dtype=np.float32)
    elif isinstance(bert_entry, dict):
        bert_emb = bert_entry['cls_embedding'].astype(np.float32)
    else:
        bert_emb = np.array(bert_entry, dtype=np.float32)

    return np.concatenate([
        np.array(lstm_emb, dtype=np.float32),
        bert_emb
    ])

# Build train embeddings
X_train_fusion = np.stack([
    get_fusion_embedding(sid) for sid in train_subjects
])
X_test_fusion = np.stack([
    get_fusion_embedding(sid) for sid in test_subjects
])

print(f"  Fusion embedding shape: {X_train_fusion.shape}  "
      f"(expect (N, 832))")
assert X_train_fusion.shape[1] == 832, \
    f"Expected 832 dims, got {X_train_fusion.shape[1]}"

# ── Dataset ───────────────────────────────────────────────────────────────────
class FusionDataset(tdata.Dataset):
    def __init__(self, X, y):
        # Pre-load to GPU for maximum training speed
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.LongTensor(y).to(device)
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ── Fusion model ──────────────────────────────────────────────────────────────
class FusionModel(nn.Module):
    """
    832-dim concatenated embedding → AKI prediction.
    Mirrors paper's architecture: FC → ReLU → Dropout → FC → ReLU → Dropout → Softmax
    """
    def __init__(self, input_dim=832, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# ── Training ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Training Multimodal Fusion Model")
print("="*60)

# Class-weighted loss for imbalanced data
n_pos = int(np.sum(y_train))
n_neg = len(y_train) - n_pos
pos_weight = torch.tensor([1.0, n_neg / n_pos]).to(device)
criterion  = nn.CrossEntropyLoss(weight=pos_weight)
print(f"  Class weight ratio: {n_neg/n_pos:.2f}x  ({n_neg} neg / {n_pos} pos)")

model     = FusionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                             weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5, verbose=False
)

# Patient-level validation split (10%)
val_n   = max(1, int(0.1 * len(train_subjects)))
train_n = len(train_subjects) - val_n

train_loader = tdata.DataLoader(
    FusionDataset(X_train_fusion[:train_n], y_train[:train_n]),
    batch_size=512,   # large batch — fusion model is tiny, GPU has headroom
    shuffle=True
)
val_loader = tdata.DataLoader(
    FusionDataset(X_train_fusion[train_n:], y_train[train_n:]),
    batch_size=512,
    shuffle=False
)

print(f"  Train: {train_n:,}  Val: {val_n:,}  "
      f"batch_size=512  early_stopping patience=10")

best_val_loss  = float('inf')
patience_count = 0
PATIENCE       = 10
EPOCHS         = 100  # will early-stop well before this

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            val_loss += criterion(model(xb), yb).item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:03d}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        patience_count = 0
        torch.save(model.state_dict(), '../models/fusion_model.pt')
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}  "
                  f"best_val_loss={best_val_loss:.4f}")
            break

# Reload best checkpoint
model.load_state_dict(
    torch.load('../models/fusion_model.pt',
               map_location=device, weights_only=True)
)
model.eval()
print("  Saved: models/fusion_model.pt")

# ── Evaluate fusion model ─────────────────────────────────────────────────────
print("\nEvaluating fusion model...")
X_test_t = torch.FloatTensor(X_test_fusion).to(device)

with torch.no_grad():
    logits_test    = model(X_test_t)
    y_prob_fusion  = torch.softmax(logits_test, dim=1)[:,1].cpu().numpy()
    y_pred_fusion  = torch.argmax(logits_test, dim=1).cpu().numpy()

def metrics(name, y_true, y_pred, y_prob):
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred,  zero_division=0)
    rec   = recall_score(y_true, y_pred,     zero_division=0)
    try:
        auc   = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auc, auprc = 0.0, 0.0
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[{name}]")
    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  AUROC={auc:.4f}  AUPRC={auprc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    return {'Model': name, 'Accuracy': round(acc,4),
            'Precision': round(prec,4), 'Recall': round(rec,4),
            'AUROC': round(auc,4), 'AUPRC': round(auprc,4)}

fusion_metrics = metrics('Multimodal Fusion',
                         y_test, y_pred_fusion, y_prob_fusion)

# ── McNemar's test vs best unimodal ──────────────────────────────────────────
print("\n" + "="*60)
print("McNemar's Test — Fusion vs Best Unimodal (LSTM)")
print("="*60)

# Reload LSTM predictions
import ast
test_df_full = pd.read_csv('../output/test_cohort.csv')

def parse_ts(s):
    try:    return np.array(ast.literal_eval(s), dtype=np.float32)
    except: return np.zeros((6,72), dtype=np.float32)

X_test_raw = np.stack(test_df_full['ts_matrix'].apply(parse_ts).values)

class TS_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=72, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(64, 2)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(self.dropout(hn[-1]))

lstm_model = TS_LSTM().to(device)
lstm_model.load_state_dict(
    torch.load('../models/lstm_model.pt',
               map_location=device, weights_only=True)
)
lstm_model.eval()

with torch.no_grad():
    X_t = torch.FloatTensor(X_test_raw).to(device)
    lstm_preds = torch.argmax(lstm_model(X_t), dim=1).cpu().numpy()

# McNemar contingency table
both_correct      = np.sum((y_pred_fusion == y_test) & (lstm_preds == y_test))
fusion_only       = np.sum((y_pred_fusion == y_test) & (lstm_preds != y_test))
lstm_only         = np.sum((y_pred_fusion != y_test) & (lstm_preds == y_test))
both_wrong        = np.sum((y_pred_fusion != y_test) & (lstm_preds != y_test))

contingency = np.array([[both_correct, fusion_only],
                        [lstm_only,    both_wrong]])

# McNemar's test
b, c = fusion_only, lstm_only
if b + c > 0:
    chi2_stat = (abs(b - c) - 1)**2 / (b + c)
    from scipy.stats import chi2
    p_value = chi2.sf(chi2_stat, df=1)
else:
    chi2_stat, p_value = 0.0, 1.0

print(f"  Contingency table:")
print(f"    Both correct:      {both_correct:,}")
print(f"    Fusion only right: {fusion_only:,}")
print(f"    LSTM only right:   {lstm_only:,}")
print(f"    Both wrong:        {both_wrong:,}")
print(f"\n  McNemar statistic: {chi2_stat:.4f}")
print(f"  p-value:           {p_value:.4f}")
print(f"  Significant (p<0.05): {p_value < 0.05}")
print(f"  Paper reported p=3.7×10⁻³ for AKI")

# ── Final comparison table ────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL FIVE-MODEL COMPARISON TABLE")
print("="*60)

# Load previous unimodal results
prev = pd.read_csv('../output/unimodal_results.csv')

paper = {
    'Logistic Regression': (0.832, 0.566),
    'XGBoost':             (0.855, 0.658),
    'LSTM':                (0.873, 0.699),
    'BioMedBERT':          (0.742, 0.420),
    'Multimodal Fusion':   (0.888, 0.727),
}

all_results = pd.concat([
    prev[['Model','Accuracy','Precision','Recall','AUROC','AUPRC']],
    pd.DataFrame([fusion_metrics])
], ignore_index=True)

all_results['Paper_AUROC'] = all_results['Model'].map(
    lambda m: paper.get(m, (None,None))[0]
)
all_results['Paper_AUPRC'] = all_results['Model'].map(
    lambda m: paper.get(m, (None,None))[1]
)

print(all_results.to_string(index=False))

print(f"\nKey finding: Multimodal AUROC {fusion_metrics['AUROC']:.4f} "
      f"vs paper 0.888")
print(f"Key finding: Multimodal AUPRC {fusion_metrics['AUPRC']:.4f} "
      f"vs paper 0.727")

multimodal_beats_lstm = fusion_metrics['AUROC'] > 0.8594
print(f"\nMultimodal > LSTM (expected): {multimodal_beats_lstm}")

all_results.to_csv('../output/fusion_results.csv', index=False)
print("\nSaved: output/fusion_results.csv")

# ── SHAP (fast — 500 patient sample only) ────────────────────────────────────
print("\n" + "="*60)
print("SHAP Analysis (500-patient sample for speed)")
print("="*60)

try:
    import shap

    # Use 500 random patients as background and explain another 500
    np.random.seed(42)
    bg_idx  = np.random.choice(len(X_train_fusion), 200, replace=False)
    exp_idx = np.random.choice(len(X_test_fusion),  500, replace=False)

    bg_data  = torch.FloatTensor(X_train_fusion[bg_idx]).to(device)
    exp_data = torch.FloatTensor(X_test_fusion[exp_idx]).to(device)

    # Wrapper for SHAP — returns positive class probability
    def model_predict(x):
        model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(x).to(device)
            return torch.softmax(model(t), dim=1)[:,1].cpu().numpy()

    explainer   = shap.KernelExplainer(model_predict,
                                       bg_data.cpu().numpy())
    shap_values = explainer.shap_values(exp_data.cpu().numpy(),
                                        nsamples=100)

    # Split SHAP values into LSTM (first 64) and BERT (next 768) components
    lstm_shap = np.abs(shap_values[:, :64]).mean(axis=0)
    bert_shap = np.abs(shap_values[:, 64:]).mean(axis=0)

    print(f"  Mean |SHAP| for LSTM component: {lstm_shap.mean():.4f}")
    print(f"  Mean |SHAP| for BERT component: {bert_shap.mean():.4f}")
    dominant = "LSTM" if lstm_shap.mean() > bert_shap.mean() else "BERT"
    print(f"  Dominant modality by SHAP:       {dominant}")
    print(f"  (Paper found time-series features dominated early)")

    with open('../embeddings/shap_values.pkl', 'wb') as f:
        pickle.dump({'shap_values': shap_values,
                     'lstm_shap':   lstm_shap,
                     'bert_shap':   bert_shap}, f)
    print("  Saved: embeddings/shap_values.pkl")

except ImportError:
    print("  SHAP not installed — skipping. Install with: pip install shap")
except Exception as e:
    print(f"  SHAP skipped: {e}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("REPLICATION COMPLETE")
print("="*60)
print(f"""
Results saved to:
  output/fusion_results.csv     ← five-model comparison table
  models/fusion_model.pt        ← trained fusion model
  embeddings/shap_values.pkl    ← SHAP analysis

IS_MOCK_BERT: {bert_data['IS_MOCK_BERT']}

Next steps:
  1. Fix BioMedBERT class weights in train_unimodal.py
     (add pos_weight to CrossEntropyLoss)
  2. Rerun train_unimodal.py to get real BERT AUROC
  3. Rerun this script for final fusion results
  4. git add output/ models/ && git commit && git push
""")