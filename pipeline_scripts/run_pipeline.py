import pandas as pd
import numpy as np
import os
from notes_preprocessing import get_tokenizer, process_patient_notes

# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------
DATA_DIR = '../data' 
OUTPUT_DIR = '../output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature Taxonomy mapping ITEMID to Feature Name
ITEMID_MAPPING = {
    # Vitals
    "Heart Rate": [220045],
    "Systolic BP": [220179, 220050],
    "Diastolic BP": [220180, 220051],
    "Mean BP": [220181, 220052],
    "Respiratory Rate": [220210],
    "Temperature (C)": [223762],
    "SpO2": [220277],
    "GCS Motor": [223901],
    "GCS Verbal": [223900],
    "GCS Eye": [220739],
    "GCS Total": [224209],
    # Labs
    "Glucose": [220621, 225664, 226537],
    "BUN": [225624],
    "Creatinine": [220615],
    "Sodium": [220645],
    "Potassium": [227442],
    "Bicarbonate": [227443],
    "Chloride": [220602],
    "Anion Gap": [227073],
    "Albumin": [227456],
    "Lactate": [225668],
    "Hemoglobin": [220228],
    "Hematocrit": [220545],
    "WBC": [220546],
    "Neutrophils": [225651], 
    "Platelets": [227457],
    "RBC": [220547], 
    "Phosphate": [225677],
    "Magnesium": [220635],
    "Calcium": [225625],
    # Fluids/Output
    "Urine Output": [226559, 226560, 226561, 226584, 226563],
    # Other
    "Weight": [226512],
    "pH": [223830],
    "PaO2": [220224],
    "PaCO2": [220235],
    "FiO2": [223835]
}

# Invert mapping for quick lookup: itemid -> feature_name
ITEMID_TO_FEATURE = {}
for feature, itemids in ITEMID_MAPPING.items():
    for itemid in itemids:
        ITEMID_TO_FEATURE[itemid] = feature

ORDERED_FEATURES = list(ITEMID_MAPPING.keys())
print(f"Total clinical features mapped: {len(ORDERED_FEATURES)}")

# ---------------------------------------------------------------------------
# 1. Cohort Selection
# ---------------------------------------------------------------------------
def extract_cohort(data_dir):
    print("--- Cohort Selection ---")
    patients = pd.read_csv(os.path.join(data_dir, 'patients.csv'))
    admits = pd.read_csv(os.path.join(data_dir, 'admissions.csv'))
    stays = pd.read_csv(os.path.join(data_dir, 'icustays.csv'))
    
    stays = stays.merge(admits[['hadm_id', 'subject_id', 'admittime', 'dischtime']], on=['subject_id', 'hadm_id'], how='inner')
    stays = stays.merge(patients[['subject_id', 'anchor_age', 'anchor_year']], on='subject_id', how='inner')
    
    stays = stays[stays['anchor_age'] >= 18]
    stay_counts = stays.groupby('hadm_id').size()
    stays = stays[stays['hadm_id'].isin(stay_counts[stay_counts == 1].index)]
    return stays

# ---------------------------------------------------------------------------
# 2. Target Generation (KDIGO)
# ---------------------------------------------------------------------------
def generate_targets(stays, data_dir):
    kdigo = pd.read_csv(os.path.join(data_dir, 'kdigo_stages.csv'))
    kdigo['charttime'] = pd.to_datetime(kdigo['charttime'])
    stays['intime'] = pd.to_datetime(stays['intime'])
    
    targets = []
    for _, row in stays.iterrows():
        label_window_start = row['intime'] + pd.Timedelta(hours=6)
        label_window_end = row['intime'] + pd.Timedelta(hours=18)
        
        mask = (kdigo['stay_id'] == row['stay_id']) & \
               (kdigo['charttime'] > label_window_start) & \
               (kdigo['charttime'] <= label_window_end)
        
        aki_target = 1 if not kdigo[mask].empty and kdigo[mask]['aki_stage'].max() >= 2 else 0
        targets.append({'stay_id': row['stay_id'], 'aki_target': aki_target, 
                        'label_window_start': label_window_start, 'label_window_end': label_window_end})
                        
    return stays.merge(pd.DataFrame(targets), on='stay_id', how='inner')

# ---------------------------------------------------------------------------
# 3. Clinical Notes Processing
# ---------------------------------------------------------------------------
def process_notes(stays, data_dir):
    print("--- Clinical Notes Processing ---")
    notes = pd.read_csv(os.path.join(data_dir, 'discharge.csv'))
    tokenizer, has_real = get_tokenizer()
    
    stays['note_tokens'] = stays.apply(lambda row: process_patient_notes(
        notes[notes['hadm_id'] == row['hadm_id']], tokenizer, has_real
    ), axis=1)
    
    return stays[stays['note_tokens'].apply(len) > 0]

# ---------------------------------------------------------------------------
# 4. Time-Series Processing (Raw Extraction)
# ---------------------------------------------------------------------------
def extract_raw_timeseries(stays, data_dir):
    print("--- Time-Series Raw Extraction ---")
    NEEDED_ITEMIDS = list(ITEMID_TO_FEATURE.keys())
    
    chunks = []
    
    # 1. Chunked reading for chartevents
    chartevents_path = os.path.join(data_dir, 'chartevents.csv.gz')
    if not os.path.exists(chartevents_path):
        chartevents_path = os.path.join(data_dir, 'chartevents.csv')
        
    if os.path.exists(chartevents_path):
        for chunk in pd.read_csv(
            chartevents_path,
            chunksize=1_000_000,
            usecols=lambda c: c in ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'valuenum']
        ):
            filtered = chunk[chunk['itemid'].isin(NEEDED_ITEMIDS)]
            chunks.append(filtered)
            
    # 2. Chunked reading for labevents
    labevents_path = os.path.join(data_dir, 'labevents.csv.gz')
    if not os.path.exists(labevents_path):
        labevents_path = os.path.join(data_dir, 'labevents.csv')
        
    if os.path.exists(labevents_path):
        for chunk in pd.read_csv(
            labevents_path,
            chunksize=1_000_000,
            usecols=lambda c: c in ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'valuenum']
        ):
            filtered = chunk[chunk['itemid'].isin(NEEDED_ITEMIDS)]
            chunks.append(filtered)

    if chunks:
        events = pd.concat(chunks, ignore_index=True)
    else:
        events = pd.DataFrame(columns=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'valuenum'])

    events['charttime'] = pd.to_datetime(events['charttime'])
    
    # Map itemids to standard features
    events['feature_name'] = events['itemid'].map(ITEMID_TO_FEATURE)
    events = events.dropna(subset=['feature_name'])
    
    raw_ts_data = {}
    raw_mask_data = {}
    for _, row in stays.iterrows():
        patient_events = events[(events['stay_id'] == row['stay_id']) & 
                                (events['charttime'] >= row['intime']) & 
                                (events['charttime'] <= row['label_window_start'])].copy()
        
        if not patient_events.empty:
            assert patient_events['charttime'].max() <= row['label_window_start'], "Leakage: Events past window"
            
        patient_events['hour'] = np.floor((patient_events['charttime'] - row['intime']) / pd.Timedelta(hours=1)).astype(int)
        
        # Aggregate duplicates within the same hour
        hourly = patient_events.pivot_table(index='hour', columns='feature_name', values='valuenum', aggfunc='mean')
        hourly = hourly.reindex(index=range(6), columns=ORDERED_FEATURES)
        
        # Compute missingness mask BEFORE imputation: 1 = missing/imputed, 0 = observed
        mask = hourly.isna().astype(float)
        
        # Forward fill imputation
        hourly = hourly.ffill()
        
        raw_ts_data[row['stay_id']] = hourly
        raw_mask_data[row['stay_id']] = mask
        
    return stays, raw_ts_data, raw_mask_data

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Pipeline...")
    cohort = extract_cohort(DATA_DIR)
    cohort = generate_targets(cohort, DATA_DIR)
    cohort = process_notes(cohort, DATA_DIR)
    cohort, raw_ts_data, raw_mask_data = extract_raw_timeseries(cohort, DATA_DIR)
    
    # Subject-level Train/Test Split BEFORE Normalization
    print("--- Train/Test Split ---")
    subjects = cohort['subject_id'].unique()
    np.random.seed(42)
    np.random.shuffle(subjects)
    split_idx = int(len(subjects) * 0.8)
    if split_idx == 0 and len(subjects) > 1: split_idx = 1
        
    train_subjects = set(subjects[:split_idx])
    test_subjects = set(subjects[split_idx:])
    
    train_cohort = cohort[cohort['subject_id'].isin(train_subjects)].copy()
    test_cohort = cohort[cohort['subject_id'].isin(test_subjects)].copy()
    
    assert len(train_subjects.intersection(test_subjects)) == 0, "Leakage in split"
    
    # -----------------------------------------------------------------------
    # STRICT NORMALIZATION: Compute stats on TRAIN ONLY
    # -----------------------------------------------------------------------
    print("--- Normalization (Train Set Statistics Only) ---")
    train_matrices = [raw_ts_data[sid] for sid in train_cohort['stay_id']]
    if train_matrices:
        global_train_df = pd.concat(train_matrices)
        train_mean = global_train_df.mean()
        train_std = global_train_df.std().replace(0, 1) # Prevent div by 0
        train_std = train_std.fillna(1)
    else:
        train_mean = pd.Series(0, index=ORDERED_FEATURES)
        train_std = pd.Series(1, index=ORDERED_FEATURES)
        
    def apply_normalization(stay_id):
        df = raw_ts_data[stay_id]
        mask = raw_mask_data[stay_id] # Use the pre-imputation mask
        
        # Normalize using TRAIN statistics
        normalized = (df - train_mean) / train_std
        # Zero-impute remaining NaNs
        normalized = normalized.fillna(0)
        
        # Concatenate normalized values and missingness masks
        combined = pd.concat([normalized, mask.add_suffix('_mask')], axis=1)
        return combined.values.tolist()

    train_cohort['ts_matrix'] = train_cohort['stay_id'].apply(apply_normalization)
    test_cohort['ts_matrix'] = test_cohort['stay_id'].apply(apply_normalization)
    
    # Verification
    sample = train_cohort.iloc[0]
    ts_shape = np.array(sample['ts_matrix']).shape
    
    print("\n--- PIPELINE VERIFICATION ---")
    print(f"Ordered Features List ({len(ORDERED_FEATURES)} items):")
    print(ORDERED_FEATURES)
    print(f"\nFinal Train Size: {len(train_cohort)}")
    print(f"Final Test Size: {len(test_cohort)}")
    print(f"\nSAMPLE OUTPUT (Subject: {sample['subject_id']}, Stay: {sample['stay_id']})")
    print(f"AKI Target: {sample['aki_target']}")
    print(f"Note Subsequences extracted: {len(sample['note_tokens'])}")
    print(f"TS Matrix Shape: {ts_shape}")
    
    # Sample Mask Fraction
    sample_mask = raw_mask_data[sample['stay_id']]
    mask_fraction = sample_mask.values.mean()
    print(f"Sample Mask Imputed Fraction (1=imputed, 0=observed): {mask_fraction:.1%}")
    
    if ts_shape[1] == len(ORDERED_FEATURES) * 2:
        print("-> SUCCESS: Matrix channels exactly match 2x feature count (value + mask).")
    
    print("Pipeline execution complete.")
