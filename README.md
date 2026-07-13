# AKI Forecasting Replication
Replication of: **"Forecasting Acute Kidney Injury and Resource Utilization 
in ICU Patients Using Longitudinal, Multimodal Models"**  
Tan et al., Journal of Biomedical Informatics, 2024  
PMID: 38692464

## Overview
This repository contains a complete replication pipeline for AKI prediction
using MIMIC-IV v2.2, implementing multimodal fusion of time-series ICU data
and clinical notes.

## Results Summary
| Model | Paper AUROC | Our AUROC | Paper AUPRC | Our AUPRC |
|---|---|---|---|---|
| Logistic Regression | 0.832 | 0.831 | 0.566 | 0.431 |
| XGBoost | 0.855 | 0.873 | 0.658 | 0.520 |
| LSTM | 0.873 | 0.859 | 0.699 | 0.496 |
| BioMedBERT | 0.742 | 0.633 | 0.420 | 0.207 |
| Multimodal Fusion | 0.888 | 0.806 | 0.727 | 0.413 |

## Dataset
- MIMIC-IV v2.2 (PhysioNet — requires credentialed access)
- MIMIC-IV-Note v2.2 (separate credentialed access required)
- Final cohort: 59,655 ICU stays
- Train: 47,623 | Test: 12,032
- AKI positive rate: 12.8% (paper: ~37% — see Differences section)

## Repository Structure
aki-replica/
├── pipeline_scripts/
│   ├── run_pipeline.py          # Cohort selection, features, labels
│   ├── generate_kdigo.py        # KDIGO 2012 AKI staging via DuckDB
│   ├── notes_preprocessing.py   # Clinical notes extraction
│   ├── train_unimodal.py        # LR, XGBoost, LSTM, BioMedBERT
│   ├── fast_bert_only.py        # BioMedBERT-only retraining (fast)
│   ├── train_fusion.py          # Multimodal fusion + evaluation
│   └── verify_deliverables.py   # Pipeline verification checks
├── output/
│   ├── train_cohort.csv         # Generated — not in repo (MIMIC data)
│   ├── test_cohort.csv          # Generated — not in repo (MIMIC data)
│   ├── unimodal_results.csv     # Model comparison table
│   └── fusion_results.csv       # Final five-model table
├── models/                      # Trained model files (not in repo)
└── embeddings/                  # Patient embeddings (not in repo)

## How to Run
1. Obtain MIMIC-IV and MIMIC-IV-Note access from PhysioNet
2. Place CSV files in pipeline_scripts/data/ (see run_pipeline.py)
3. Install dependencies: pip install -r requirements.txt
4. Run in order:
   python generate_kdigo.py --data-root data
   python run_pipeline.py
   python train_unimodal.py
   python fast_bert_only.py    # if BERT needs retraining only
   python train_fusion.py

## Key Differences from Paper
1. KDIGO positive rate: 12.8% vs paper's ~37%
   - We used strict KDIGO 2012 dynamic rolling baselines
   - Paper likely used precomputed BigQuery MIMIC-IV derived table
2. Feature count: 36 vs ~50 (paper's list not published)
3. MIMIC-IV v2.2 vs paper's version (4.8% larger cohort)
4. Patient-level train/test split (stricter than paper's event-level)

## Dependencies
See requirements.txt

## Citation
Tan Y, Dede M, Mohanty V, et al. Forecasting Acute Kidney Injury and 
Resource Utilization in ICU Patients Using Longitudinal, Multimodal Models.
J Biomed Inform. 2024;154:104648. doi:10.1016/j.jbi.2024.104648
