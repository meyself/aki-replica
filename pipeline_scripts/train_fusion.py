import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
from statsmodels.stats.contingency_tables import mcnemar
import shap
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# MANDATORY FIRST STEP — IS_MOCK_BERT Check
with open('../embeddings/biomedbert_cls_embeddings.pkl', 'rb') as f:
    bert_emb_data = pickle.load(f)

if bert_emb_data.get('IS_MOCK_BERT', False):
    print("WARNING: Running fusion on MOCK BioMedBERT embeddings.")
    print("Fusion metrics are structurally verified but numerically meaningless.")
    print("Before presenting results, regenerate real embeddings by running")
    print("train_unimodal.py on a machine with HuggingFace internet access.\n")

# Dummy Data matching train_unimodal.py
train_subjects = [99901, 99902]
test_subjects = [99903, 99904]
y_train = np.array([0, 1])
y_test = np.array([0, 1])
y_tr_t = torch.LongTensor(y_train)
y_te_t = torch.LongTensor(y_test)

with open('../embeddings/lstm_hidden_states.pkl', 'rb') as f:
    lstm_emb = pickle.load(f)

bert_embeddings = bert_emb_data['embeddings']

# Create Fusion tensors
X_train_fusion = []
for subj in train_subjects:
    X_train_fusion.append(np.concatenate([lstm_emb[subj], bert_embeddings[subj]]))
X_train_fusion = torch.FloatTensor(np.array(X_train_fusion))

X_test_fusion = []
for subj in test_subjects:
    X_test_fusion.append(np.concatenate([lstm_emb[subj], bert_embeddings[subj]]))
X_test_fusion = torch.FloatTensor(np.array(X_test_fusion))

# Model architecture
class MultimodalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(832, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

fusion_model = MultimodalFusion()
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

if len(train_subjects) >= 10:
    val_size = max(1, int(0.1 * len(train_subjects)))
else:
    print("WARNING: Train set too small for early stopping. Training for full 50 epochs.")

for epoch in range(50):
    fusion_model.train()
    optimizer.zero_grad()
    logits = fusion_model(X_train_fusion)
    loss = criterion(logits, y_tr_t)
    loss.backward()
    optimizer.step()

fusion_model_path = '../models/fusion_model.pt'
torch.save(fusion_model.state_dict(), fusion_model_path)
print(f"\nSaved fusion model to {fusion_model_path}")
print(f"models/fusion_model.pt -> EXISTS: {os.path.exists(fusion_model_path)}\n")

# Eval Multimodal
fusion_model.eval()
with torch.no_grad():
    logits = fusion_model(X_test_fusion)
    y_prob_fusion = torch.softmax(logits, dim=1)[:, 1].numpy()
    y_pred_fusion = torch.argmax(logits, dim=1).numpy()

# ---------------------------------------------------------
# Re-evaluate unimodal models to build the table
# ---------------------------------------------------------
X_test_raw = np.random.randn(2, 6, 72)
X_test_eng = []
for matrix in X_test_raw:
    flattened = matrix.flatten()
    values = matrix[:, :36]
    masks = matrix[:, 36:]
    min_vals = np.min(values, axis=0)
    max_vals = np.max(values, axis=0)
    mean_vals = np.mean(values, axis=0)
    std_vals = np.std(values, axis=0)
    n = values.shape[0]
    m3 = np.sum((values - mean_vals)**3, axis=0) / n
    m2 = np.sum((values - mean_vals)**2, axis=0) / n
    skew_vals = m3 / (m2**1.5 + 1e-8)
    skew_vals = np.nan_to_num(skew_vals, 0)
    count_observed = np.sum(masks == 0, axis=0)
    eng = np.concatenate([min_vals, max_vals, mean_vals, std_vals, skew_vals, count_observed])
    X_test_eng.append(np.concatenate([flattened, eng]))
X_test_eng = np.array(X_test_eng)

with open('../models/lr_model.pkl', 'rb') as f:
    lr = pickle.load(f)
y_pred_lr = lr.predict(X_test_eng)
y_prob_lr = lr.predict_proba(X_test_eng)[:, 1]

with open('../models/xgb_model.pkl', 'rb') as f:
    xgb = pickle.load(f)
y_pred_xgb = xgb.predict(X_test_eng)
y_prob_xgb = xgb.predict_proba(X_test_eng)[:, 1]

class TS_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=72, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        hidden = hn[-1]
        drop = self.dropout(hidden)
        logits = self.fc(drop)
        return logits, hidden

lstm = TS_LSTM()
lstm.load_state_dict(torch.load('../models/lstm_model.pt', weights_only=True))
lstm.eval()
with torch.no_grad():
    logits, _ = lstm(torch.FloatTensor(X_test_raw))
    y_prob_lstm = torch.softmax(logits, dim=1)[:, 1].numpy()
    y_pred_lstm = torch.argmax(logits, dim=1).numpy()

# BioMedBERT Mock — class must be defined here so pickle can deserialize it
class DummyBioMedBERT:
    def get_predictions_and_embeddings(self, note_chunks):
        n = len(note_chunks)
        if n == 0:
            return 0.5, np.zeros(768)
        probs = np.random.uniform(0.1, 0.9, size=n)
        P_max = np.max(probs)
        P_mean = np.mean(probs)
        c = 2
        P_final = (P_max + (n/c) * P_mean) / (1 + n/c)
        embedding = np.random.randn(768)
        return P_final, embedding

with open('../models/biomedbert_embeddings.pkl', 'rb') as f:
    bert_model = pickle.load(f)
y_prob_bert = []
y_pred_bert = []
test_notes = [[101, 400, 102], [101, 500, 102]]
for notes in test_notes:
    p_final, _ = bert_model.get_predictions_and_embeddings(notes)
    y_prob_bert.append(p_final)
    y_pred_bert.append(1 if p_final > 0.5 else 0)

# Build Table
results = []
def get_metrics(name, y_true, y_pred, y_prob):
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'AUROC': roc_auc_score(y_true, y_prob),
        'AUPRC': average_precision_score(y_true, y_prob)
    }

results.append(get_metrics('LR', y_test, y_pred_lr, y_prob_lr))
results.append(get_metrics('XGBoost', y_test, y_pred_xgb, y_prob_xgb))
results.append(get_metrics('LSTM', y_test, y_pred_lstm, y_prob_lstm))
results.append(get_metrics('BioMedBERT', y_test, y_pred_bert, y_prob_bert))
results.append(get_metrics('Multimodal', y_test, y_pred_fusion, y_prob_fusion))

df = pd.DataFrame(results).round(3)
df['Paper_AUROC_Ref'] = [0.832, 0.855, 0.873, 0.742, 0.888]

print("--- FINAL COMPARISON TABLE ---")
print(df.to_string(index=False))

# McNemar's Test
print("\n--- MCNEMAR'S TEST ---")
print("Note: With N=2 test patients, the test result is statistically meaningless.")
print("This is just verifying the code runs. The paper reported p=3.7x10^-3 for AKI.")
# Compare Fusion vs Best Unimodal (say, LSTM based on AUROC in paper)
c00 = sum((y_pred_fusion == y_test) & (y_pred_lstm == y_test))
c01 = sum((y_pred_fusion == y_test) & (y_pred_lstm != y_test))
c10 = sum((y_pred_fusion != y_test) & (y_pred_lstm == y_test))
c11 = sum((y_pred_fusion != y_test) & (y_pred_lstm != y_test))
table = [[c00, c01], [c10, c11]]
result = mcnemar(table, exact=True)
print(f"Test Statistic: {result.statistic}")
print(f"p-value: {result.pvalue}")

# SHAP Analysis — using manual gradient-based saliency (no TF dependency)
print("\n--- SHAP ANALYSIS (LSTM Component) ---")
print("Note: SHAP on random mock data is expected to produce arbitrary feature rankings.")
print("Paper's top features for AKI: urine output, SpO2, diastolic BP, RBC count, weight.")

try:
    features = [
        'Heart Rate', 'Systolic BP', 'Diastolic BP', 'Mean BP', 'Respiratory Rate',
        'Temperature (C)', 'SpO2', 'GCS Motor', 'GCS Verbal', 'GCS Eye', 'GCS Total',
        'Glucose', 'BUN', 'Creatinine', 'Sodium', 'Potassium', 'Bicarbonate', 'Chloride',
        'Anion Gap', 'Albumin', 'Lactate', 'Hemoglobin', 'Hematocrit', 'WBC', 'Neutrophils',
        'Platelets', 'RBC', 'Phosphate', 'Magnesium', 'Calcium', 'Urine Output',
        'Weight', 'pH', 'PaO2', 'PaCO2', 'FiO2'
    ]

    lstm.eval()
    X_test_tensor = torch.FloatTensor(X_test_raw)
    X_test_tensor.requires_grad_(True)

    # Forward pass: get class-1 logit
    logits_shap, _ = lstm(X_test_tensor)
    score = logits_shap[:, 1].sum()
    score.backward()

    # Gradient shape: (N, 6, 72). Use abs mean over patients as attribution proxy.
    grad = X_test_tensor.grad.detach().numpy()  # (2, 6, 72)
    mean_abs_grad = np.mean(np.abs(grad), axis=0)  # (6, 72)

    shap_output = {}
    for t in range(6):
        timestep_grad = mean_abs_grad[t, :36]  # only clinical features, not masks
        top_indices = np.argsort(timestep_grad)[::-1][:5]
        top_names = [features[i] for i in top_indices]
        shap_output[f'timestep_{t}'] = top_names
        print(f"  Timestep {t} top 5 features: {', '.join(top_names)}")

    # Check overlap with paper's top features
    paper_top = {'Urine Output', 'SpO2', 'Diastolic BP', 'RBC', 'Weight'}
    all_mock_top = set(f for v in shap_output.values() for f in v)
    overlap = paper_top.intersection(all_mock_top)
    if overlap:
        print(f"\n  Overlap with paper's top features on mock data: {overlap}")
        print("  (Coincidental on random data — meaningful overlap expected only on real MIMIC-IV)")
    else:
        print("\n  No overlap with paper's top features. Expected: SHAP on random mock data produces arbitrary rankings.")

    shap_save = {'IS_MOCK_FUSION': True, 'gradient_attributions': mean_abs_grad, 'per_timestep_top5': shap_output}
    with open('../embeddings/shap_values.pkl', 'wb') as f:
        pickle.dump(shap_save, f)
    print("\n  Saved SHAP/gradient attribution values to embeddings/shap_values.pkl")

except Exception as e:
    print(f"  SHAP Analysis failed: {e}")

