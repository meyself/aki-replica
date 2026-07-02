import os
import pickle
import numpy as np

print("--- CHECK 2: Confirming model files exist on disk ---")
model_files = [
    '../models/lr_model.pkl',
    '../models/xgb_model.pkl',
    '../models/lstm_model.pt',
    '../models/biomedbert_embeddings.pkl'
]
for f in model_files:
    print(f, '-> EXISTS:', os.path.exists(f))

print("\n--- CHECK 3: Confirming embedding shapes ---")
with open('../embeddings/lstm_hidden_states.pkl', 'rb') as f:
    lstm_emb = pickle.load(f)
with open('../embeddings/biomedbert_cls_embeddings.pkl', 'rb') as f:
    bert_emb = pickle.load(f)

print('LSTM embeddings shape:', np.array(list(lstm_emb.values())).shape)
print('BioMedBERT embeddings shape:', np.array(list(bert_emb['embeddings'].values())).shape)
print('IS_MOCK_BERT flag:', bert_emb['IS_MOCK_BERT'])
