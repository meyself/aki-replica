"""
fast_bert_only.py — Stable BioMedBERT fine-tuning for AKI prediction

Two-stage training protocol (standard for BERT fine-tuning with class imbalance):
  Stage 1 (2 epochs): Freeze BERT backbone — only train the classification head
                      Higher LR (1e-3), stronger class weight (4x)
                      Head learns the task without disturbing pretrained weights
  Stage 2 (2 epochs): Unfreeze full model — fine-tune everything together
                      Lower LR (2e-5), moderate class weight (2x)
                      Backbone adapts to clinical AKI language

Also uses:
  - Linear warmup LR scheduler (prevents early overshooting)
  - Gradient clipping (max_grad_norm=1.0) (prevents oscillation)
  - 256 tokens (4x faster than 512)
  - Batched embedding extraction (10-20x faster than one patient at a time)

Expected runtime: 60-90 minutes total
Expected BioMedBERT AUROC: 0.58-0.70 (paper: 0.742 at 37% positive rate)
"""

import os
import ast
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as tdata
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, precision_score,
                             recall_score, confusion_matrix)
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_TOKENS             = 256
MAX_CHUNKS_PER_PATIENT = 3
BERT_BATCH_TRAIN       = 32
BERT_BATCH_EXTRACT     = 64
BERT_MODEL_NAME = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'

# Two-stage training config
STAGE1_EPOCHS      = 2      # frozen backbone — head only
STAGE1_LR          = 1e-3   # higher LR for head-only training
STAGE1_POS_WEIGHT  = 4.0    # moderate weight for head stabilization

STAGE2_EPOCHS      = 2      # full model fine-tuning
STAGE2_LR          = 2e-5   # standard BERT fine-tuning LR
STAGE2_POS_WEIGHT  = 2.0    # lower weight once head is stable

WARMUP_RATIO       = 0.1    # 10% of steps for LR warmup
MAX_GRAD_NORM      = 1.0    # gradient clipping threshold

os.makedirs('../models',     exist_ok=True)
os.makedirs('../embeddings', exist_ok=True)
os.makedirs('../output',     exist_ok=True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Load cohort
# ---------------------------------------------------------------------------
print("\nLoading cohort data...")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

train_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_cohort.csv'))
test_df  = pd.read_csv(os.path.join(OUTPUT_DIR, 'test_cohort.csv'))

def parse_notes(s):
    if pd.isna(s) or s == '[]': return []
    try:    return ast.literal_eval(str(s))
    except: return []

train_df['notes_parsed'] = train_df['note_tokens'].apply(parse_notes)
test_df['notes_parsed']  = test_df['note_tokens'].apply(parse_notes)

train_subjects = train_df['subject_id'].tolist()
test_subjects  = test_df['subject_id'].tolist()
y_train        = train_df['aki_target'].values.astype(int)
y_test         = test_df['aki_target'].values.astype(int)
train_notes    = train_df['notes_parsed'].tolist()
test_notes     = test_df['notes_parsed'].tolist()

n_pos = int(np.sum(y_train))
n_neg = len(y_train) - n_pos
print(f"  Train: {len(train_subjects):,}  "
      f"({n_pos:,} pos / {n_neg:,} neg = {n_pos/len(y_train):.1%})")
print(f"  Test:  {len(test_subjects):,}")

# ---------------------------------------------------------------------------
# Load BioMedBERT
# ---------------------------------------------------------------------------
print("\nLoading BioMedBERT...")
import warnings
warnings.filterwarnings('ignore')

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_backbone  = AutoModel.from_pretrained(BERT_MODEL_NAME,
                                           use_safetensors=True)
print("  Loaded successfully")

class BioMedBERTClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone   = backbone
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
        # Initialize classifier head with small weights for stability
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        out     = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)
        logits  = self.classifier(cls_emb)
        return logits, cls_emb

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  Backbone FROZEN — training head only")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("  Backbone UNFROZEN — full model training")

bert_clf = BioMedBERTClassifier(bert_backbone).to(device)
print(f"  On device: {next(bert_clf.parameters()).device}")

# ---------------------------------------------------------------------------
# Build training pairs and pre-tokenize
# ---------------------------------------------------------------------------
print("\nBuilding training pairs...")
bert_train_pairs = []
for subj_idx, chunks in enumerate(train_notes):
    label = int(y_train[subj_idx])
    for chunk in chunks[:MAX_CHUNKS_PER_PATIENT]:
        if isinstance(chunk, list):
            chunk = ' '.join(str(t) for t in chunk)
        if isinstance(chunk, str) and len(chunk.strip()) > 10:
            bert_train_pairs.append((chunk, label))

print(f"  Training chunks: {len(bert_train_pairs):,}")
print(f"  Positive chunks: "
      f"{sum(1 for _,l in bert_train_pairs if l==1):,}  "
      f"({sum(1 for _,l in bert_train_pairs if l==1)/len(bert_train_pairs):.1%})")

class PreTokenizedDataset(tdata.Dataset):
    def __init__(self, pairs, tokenizer, max_len=MAX_TOKENS):
        print(f"  Pre-tokenizing {len(pairs):,} chunks...")
        texts  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]
        BATCH  = 1024
        all_ids, all_masks = [], []
        for i in range(0, len(texts), BATCH):
            enc = tokenizer(
                texts[i:i+BATCH], max_length=max_len,
                truncation=True, padding='max_length', return_tensors='pt'
            )
            all_ids.append(enc['input_ids'])
            all_masks.append(enc['attention_mask'])
        self.input_ids      = torch.cat(all_ids,   dim=0)
        self.attention_mask = torch.cat(all_masks, dim=0)
        self.labels         = torch.tensor(labels, dtype=torch.long)
        print(f"  Done — shape: {self.input_ids.shape}")

    def __len__(self):        return len(self.labels)
    def __getitem__(self, i):
        return self.input_ids[i], self.attention_mask[i], self.labels[i]

print("\nPre-tokenizing...")
dataset     = PreTokenizedDataset(bert_train_pairs, bert_tokenizer)
train_loader = tdata.DataLoader(
    dataset, batch_size=BERT_BATCH_TRAIN,
    shuffle=True, num_workers=0, pin_memory=True
)
total_steps_s1 = len(train_loader) * STAGE1_EPOCHS
total_steps_s2 = len(train_loader) * STAGE2_EPOCHS

# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, criterion,
              scheduler=None, grad_clip=None, desc=""):
    model.train()
    total_loss, n_batches = 0.0, 0
    for input_ids, attention_mask, labels in loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
        n_batches  += 1
    avg = total_loss / max(n_batches, 1)
    print(f"  {desc}  loss={avg:.4f}")
    return avg

# ---------------------------------------------------------------------------
# Stage 1 — Freeze backbone, train head only
# ---------------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"Stage 1: Head-only training ({STAGE1_EPOCHS} epochs)")
print(f"  LR={STAGE1_LR}  pos_weight={STAGE1_POS_WEIGHT}x")
print(f"{'='*55}")

bert_clf.freeze_backbone()

criterion_s1 = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, STAGE1_POS_WEIGHT]).to(device)
)
optimizer_s1 = torch.optim.Adam(
    filter(lambda p: p.requires_grad, bert_clf.parameters()),
    lr=STAGE1_LR
)
scheduler_s1 = get_linear_schedule_with_warmup(
    optimizer_s1,
    num_warmup_steps=max(1, int(total_steps_s1 * WARMUP_RATIO)),
    num_training_steps=total_steps_s1
)

for epoch in range(STAGE1_EPOCHS):
    run_epoch(bert_clf, train_loader, optimizer_s1, criterion_s1,
              scheduler_s1, grad_clip=MAX_GRAD_NORM,
              desc=f"Stage1 Epoch {epoch+1}/{STAGE1_EPOCHS}")

# Quick check after stage 1
bert_clf.eval()
sample_ids   = dataset.input_ids[:256].to(device)
sample_mask  = dataset.attention_mask[:256].to(device)
sample_labels = dataset.labels[:256].numpy()
with torch.no_grad():
    logits, _ = bert_clf(sample_ids, sample_mask)
    preds = torch.argmax(logits, dim=1).cpu().numpy()
n_pred_pos = preds.sum()
print(f"  Stage 1 sanity check on 256 chunks: "
      f"{n_pred_pos} predicted positive "
      f"({n_pred_pos/256:.1%})  "
      f"[true rate: {sample_labels.mean():.1%}]")

# ---------------------------------------------------------------------------
# Stage 2 — Unfreeze backbone, full fine-tuning
# ---------------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"Stage 2: Full model fine-tuning ({STAGE2_EPOCHS} epochs)")
print(f"  LR={STAGE2_LR}  pos_weight={STAGE2_POS_WEIGHT}x  "
      f"grad_clip={MAX_GRAD_NORM}")
print(f"{'='*55}")

bert_clf.unfreeze_backbone()

criterion_s2 = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, STAGE2_POS_WEIGHT]).to(device)
)
# Layer-wise LR: backbone gets 10x lower LR than head
optimizer_s2 = torch.optim.AdamW([
    {'params': bert_clf.backbone.parameters(),
     'lr': STAGE2_LR},
    {'params': list(bert_clf.dropout.parameters()) +
               list(bert_clf.classifier.parameters()),
     'lr': STAGE2_LR * 10}
], weight_decay=0.01)

scheduler_s2 = get_linear_schedule_with_warmup(
    optimizer_s2,
    num_warmup_steps=max(1, int(total_steps_s2 * WARMUP_RATIO)),
    num_training_steps=total_steps_s2
)

best_loss = float('inf')
for epoch in range(STAGE2_EPOCHS):
    loss = run_epoch(bert_clf, train_loader, optimizer_s2, criterion_s2,
                     scheduler_s2, grad_clip=MAX_GRAD_NORM,
                     desc=f"Stage2 Epoch {epoch+1}/{STAGE2_EPOCHS}")
    if loss < best_loss:
        best_loss = loss
        torch.save(bert_clf.state_dict(), '../models/biomedbert_finetuned.pt')

print(f"  Best loss: {best_loss:.4f}")
print("  Saved: models/biomedbert_finetuned.pt")

# Load best checkpoint
bert_clf.load_state_dict(
    torch.load('../models/biomedbert_finetuned.pt',
               map_location=device, weights_only=True)
)

# ---------------------------------------------------------------------------
# Batched embedding extraction
# ---------------------------------------------------------------------------
print(f"\nExtracting embeddings (batched, batch_size={BERT_BATCH_EXTRACT})...")

def extract_embeddings_batched(subjects, notes_list, label=""):
    all_chunks     = []
    patient_ranges = {}

    for subj_id, chunks in zip(subjects, notes_list):
        start = len(all_chunks)
        for chunk in chunks[:MAX_CHUNKS_PER_PATIENT]:
            if isinstance(chunk, list):
                chunk = ' '.join(str(t) for t in chunk)
            if isinstance(chunk, str) and len(chunk.strip()) > 10:
                all_chunks.append(chunk)
        patient_ranges[subj_id] = (start, len(all_chunks))

    print(f"  {label}: {len(subjects):,} patients, "
          f"{len(all_chunks):,} chunks")

    if not all_chunks:
        return {sid: {'cls_embedding': np.zeros(768, dtype=np.float32),
                      'P_final': 0.5}
                for sid in subjects}

    # Tokenize all chunks
    print(f"  Tokenizing...")
    TBATCH = 1024
    all_ids_list, all_masks_list = [], []
    for i in range(0, len(all_chunks), TBATCH):
        enc = bert_tokenizer(
            all_chunks[i:i+TBATCH], max_length=MAX_TOKENS,
            truncation=True, padding='max_length', return_tensors='pt'
        )
        all_ids_list.append(enc['input_ids'])
        all_masks_list.append(enc['attention_mask'])

    all_input_ids      = torch.cat(all_ids_list,   dim=0)
    all_attention_mask = torch.cat(all_masks_list, dim=0)

    # GPU forward pass in batches
    print(f"  GPU forward pass ({len(all_chunks):,} chunks)...")
    bert_clf.eval()
    all_cls_np   = np.zeros((len(all_chunks), 768), dtype=np.float32)
    all_probs_np = np.zeros(len(all_chunks), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(all_chunks), BERT_BATCH_EXTRACT):
            ids  = all_input_ids[i:i+BERT_BATCH_EXTRACT].to(device)
            mask = all_attention_mask[i:i+BERT_BATCH_EXTRACT].to(device)
            logits, cls_emb = bert_clf(ids, mask)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            cls   = cls_emb.cpu().numpy()
            end   = min(i + BERT_BATCH_EXTRACT, len(all_chunks))
            all_cls_np[i:end]   = cls
            all_probs_np[i:end] = probs
            if (i // BERT_BATCH_EXTRACT + 1) % 50 == 0:
                print(f"    {end:,} / {len(all_chunks):,} done...")

    # Aggregate per patient using paper's pooling formula
    results = {}
    for subj_id in subjects:
        start, end = patient_ranges.get(subj_id, (0, 0))
        if start == end:
            results[subj_id] = {
                'cls_embedding': np.zeros(768, dtype=np.float32),
                'P_final': 0.5
            }
            continue
        p_chunk = all_probs_np[start:end]
        c_chunk = all_cls_np[start:end]
        n, c    = len(p_chunk), 2
        # Paper's pooling: P = (P_max + (n/c)*P_mean) / (1 + n/c)
        P_final = (float(p_chunk.max()) +
                   (n/c) * float(p_chunk.mean())) / (1 + n/c)
        results[subj_id] = {
            'cls_embedding': c_chunk.mean(axis=0),
            'P_final':       float(P_final)
        }
    return results

all_subjects       = list(train_subjects) + list(test_subjects)
all_notes_combined = list(train_notes)    + list(test_notes)

embeddings = extract_embeddings_batched(
    all_subjects, all_notes_combined, label="All patients"
)

bert_embeddings = {'embeddings': embeddings, 'IS_MOCK_BERT': False}
with open('../embeddings/biomedbert_cls_embeddings.pkl', 'wb') as f:
    pickle.dump(bert_embeddings, f)
print("  Saved: embeddings/biomedbert_cls_embeddings.pkl  "
      "(IS_MOCK_BERT=False)")

# ---------------------------------------------------------------------------
# Evaluate on test set
# ---------------------------------------------------------------------------
print("\nEvaluating BioMedBERT on test set...")
y_prob_bert = [embeddings[sid]['P_final'] for sid in test_subjects]
y_pred_bert = [1 if p > 0.5 else 0 for p in y_prob_bert]

try:
    auroc = roc_auc_score(y_test, y_prob_bert)
    auprc = average_precision_score(y_test, y_prob_bert)
except ValueError:
    auroc, auprc = 0.0, 0.0

acc  = accuracy_score(y_test,  y_pred_bert)
prec = precision_score(y_test, y_pred_bert, zero_division=0)
rec  = recall_score(y_test,    y_pred_bert, zero_division=0)
cm   = confusion_matrix(y_test, y_pred_bert)

print(f"\n[BioMedBERT — Two-Stage Fine-Tuning]")
print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  "
      f"Recall={rec:.4f}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
print(f"  Confusion Matrix:\n{cm}")
print(f"\n  Paper reference:  AUROC=0.742  AUPRC=0.420")
print(f"  Your result:      AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

if auroc >= 0.65:
    print(f"  ✓ Good result — proceed to train_fusion.py")
elif auroc >= 0.55:
    print(f"  ✓ Acceptable — CLS embeddings are meaningful for fusion")
    print(f"    Note: lower AUROC expected due to 12.8% vs paper's 37% "
          f"positive rate")
else:
    print(f"  ⚠ AUROC below 0.55 — CLS embeddings may still help fusion")
    print(f"    The fusion model uses 768-dim CLS embeddings directly,")
    print(f"    not P_final, so fusion can still improve over unimodal LSTM")

# Update unimodal results
try:
    prev = pd.read_csv('../output/unimodal_results.csv')
    mask = prev['Model'].str.contains('BioMedBERT')
    prev.loc[mask, 'Accuracy']  = round(acc,   4)
    prev.loc[mask, 'Precision'] = round(prec,  4)
    prev.loc[mask, 'Recall']    = round(rec,   4)
    prev.loc[mask, 'AUROC']     = round(auroc, 4)
    prev.loc[mask, 'AUPRC']     = round(auprc, 4)
    prev.to_csv('../output/unimodal_results.csv', index=False)
    print("\n  Updated: output/unimodal_results.csv")
except Exception as e:
    print(f"\n  Note: could not update unimodal_results.csv: {e}")

print(f"""
{'='*55}
DONE
{'='*55}
IS_MOCK_BERT: False
Stage 1 ({STAGE1_EPOCHS} epochs frozen):  complete
Stage 2 ({STAGE2_EPOCHS} epochs full):    complete

Next step:
  python train_fusion.py
""")