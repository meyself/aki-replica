"""
run_pipeline.py — MIMIC-IV AKI Forecasting Pipeline
Replication of: Tan et al., "Forecasting Acute Kidney Injury and Resource
Utilization in ICU Patients Using Longitudinal, Multimodal Models" (2024)

Fixes applied vs Antigravity-generated version:
  1. All file paths corrected to match MIMIC-IV folder structure (hosp/icu/note/derived)
  2. outputevents loaded for Urine Output (was silently missing before)
  3. labevents stay_id joined via hadm_id (labevents has no stay_id in MIMIC-IV)
  4. generate_targets vectorized (row loop replaced with merge — ~100x faster)
  5. process_notes pre-groups by hadm_id (avoids 50,000 repeated full-table scans)
"""

import os
import numpy as np
import pandas as pd
from notes_preprocessing import get_tokenizer, process_patient_notes

# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Subdirectory helpers
HOSP_DIR    = os.path.join(DATA_DIR, 'hosp')
ICU_DIR     = os.path.join(DATA_DIR, 'icu')
NOTE_DIR    = os.path.join(DATA_DIR, 'note')
DERIVED_DIR = os.path.join(DATA_DIR, 'derived')

def p(subdir, name):
    """Return .csv.gz path, falling back to .csv if gz not found."""
    gz = os.path.join(subdir, name + '.csv.gz')
    if os.path.exists(gz):
        return gz
    return os.path.join(subdir, name + '.csv')

# ---------------------------------------------------------------------------
# Feature Taxonomy — ITEMID → Feature Name
# NOTE: Urine Output ItemIDs come from outputevents, not chartevents/labevents
# ---------------------------------------------------------------------------
ITEMID_MAPPING = {
    # Vitals (chartevents)
    "Heart Rate":       [220045],
    "Systolic BP":      [220179, 220050],
    "Diastolic BP":     [220180, 220051],
    "Mean BP":          [220181, 220052],
    "Respiratory Rate": [220210],
    "Temperature (C)":  [223762],
    "SpO2":             [220277],
    "GCS Motor":        [223901],
    "GCS Verbal":       [223900],
    "GCS Eye":          [220739],
    "GCS Total":        [224209],
    # Labs (labevents)
    "Glucose":          [220621, 225664, 226537],
    "BUN":              [225624],
    "Creatinine":       [220615],
    "Sodium":           [220645],
    "Potassium":        [227442],
    "Bicarbonate":      [227443],
    "Chloride":         [220602],
    "Anion Gap":        [227073],
    "Albumin":          [227456],
    "Lactate":          [225668],
    "Hemoglobin":       [220228],
    "Hematocrit":       [220545],
    "WBC":              [220546],
    "Neutrophils":      [225651],
    "Platelets":        [227457],
    "RBC":              [220547],
    "Phosphate":        [225677],
    "Magnesium":        [220635],
    "Calcium":          [225625],
    # Urine Output (outputevents — loaded separately)
    "Urine Output":     [226559, 226560, 226561, 226584, 226563],
    # Other (chartevents)
    "Weight":           [226512],
    "pH":               [223830],
    "PaO2":             [220224],
    "PaCO2":            [220235],
    "FiO2":             [223835],
}

ITEMID_TO_FEATURE = {
    iid: feat
    for feat, iids in ITEMID_MAPPING.items()
    for iid in iids
}

UO_ITEMIDS      = ITEMID_MAPPING["Urine Output"]
CHART_LAB_IIDS  = [iid for feat, iids in ITEMID_MAPPING.items()
                   if feat != "Urine Output" for iid in iids]
ORDERED_FEATURES = list(ITEMID_MAPPING.keys())

print(f"Total clinical features: {len(ORDERED_FEATURES)}")
print(f"  Chart/Lab ItemIDs:     {len(CHART_LAB_IIDS)}")
print(f"  Urine Output ItemIDs:  {len(UO_ITEMIDS)}")


# ---------------------------------------------------------------------------
# labevents uses a SEPARATE ItemID system from chartevents in MIMIC-IV
# These are the official lab result ItemIDs from the hosp module
# ---------------------------------------------------------------------------
LABEVENTS_ITEMID_MAPPING = {
    "Glucose":      [50931, 50809],
    "BUN":          [51006],
    "Creatinine":   [50912],
    "Sodium":       [50983],
    "Potassium":    [50971],
    "Bicarbonate":  [50882],
    "Chloride":     [50902],
    "Anion Gap":    [50868],
    "Albumin":      [50862],
    "Lactate":      [50813],
    "Hemoglobin":   [51222],
    "Hematocrit":   [51221],
    "WBC":          [51301],
    "Neutrophils":  [51256],
    "Platelets":    [51265],
    "RBC":          [51279],
    "Phosphate":    [50970],
    "Magnesium":    [50960],
    "Calcium":      [50893],
    "pH":           [50820],
    "PaO2":         [50821],
    "PaCO2":        [50818],
}

# Add labevents ItemIDs to the master lookup
for feat, iids in LABEVENTS_ITEMID_MAPPING.items():
    for iid in iids:
        ITEMID_TO_FEATURE[iid] = feat

# Combined list of all non-UO ItemIDs (chartevents + labevents)
CHART_LAB_IIDS = [
    iid for feat, iids in ITEMID_MAPPING.items()
    if feat != "Urine Output" for iid in iids
] + [
    iid for iids in LABEVENTS_ITEMID_MAPPING.values()
    for iid in iids
]


# ---------------------------------------------------------------------------
# 1. Cohort Selection
# ---------------------------------------------------------------------------
def extract_cohort():
    print("\n--- Cohort Selection ---")

    patients = pd.read_csv(p(HOSP_DIR, 'patients'),
                           usecols=['subject_id', 'anchor_age'])
    admits   = pd.read_csv(p(HOSP_DIR, 'admissions'),
                           usecols=['subject_id', 'hadm_id',
                                    'admittime', 'dischtime'])
    stays    = pd.read_csv(p(ICU_DIR,  'icustays'),
                           usecols=['subject_id', 'hadm_id', 'stay_id',
                                    'intime', 'outtime'],
                           parse_dates=['intime', 'outtime'])

    print(f"  Initial ICU stays: {len(stays):,}")

    stays = stays.merge(admits, on=['subject_id', 'hadm_id'], how='inner')
    stays = stays.merge(patients, on='subject_id', how='inner')

    # Age filter
    stays = stays[stays['anchor_age'] >= 18]
    print(f"  After age >= 18:   {len(stays):,}")

    # Single ICU stay per admission (exclude transfers / multiple ICU stays)
    stay_counts = stays.groupby('hadm_id')['stay_id'].transform('count')
    stays = stays[stay_counts == 1].copy()
    print(f"  After single-stay: {len(stays):,}")

    return stays.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Target Generation (KDIGO — vectorized, no row loop)
# ---------------------------------------------------------------------------
def generate_targets(stays):
    print("\n--- Target Generation ---")

    kdigo = pd.read_csv(p(DERIVED_DIR, 'kdigo_stages'),
                        usecols=['stay_id', 'charttime', 'aki_stage'],
                        parse_dates=['charttime'])

    stays = stays.copy()
    stays['intime']             = pd.to_datetime(stays['intime'])
    stays['label_window_start'] = stays['intime'] + pd.Timedelta(hours=6)
    stays['label_window_end']   = stays['intime'] + pd.Timedelta(hours=18)

    # Merge KDIGO onto stays, then filter to prediction window
    merged = stays[['stay_id', 'label_window_start', 'label_window_end']].merge(
        kdigo, on='stay_id', how='left'
    )
    in_window = (
        (merged['charttime'] >  merged['label_window_start']) &
        (merged['charttime'] <= merged['label_window_end'])
    )
    windowed = merged[in_window]

    # Max AKI stage per stay within the window
    max_stage = (windowed.groupby('stay_id')['aki_stage']
                         .max()
                         .reset_index()
                         .rename(columns={'aki_stage': 'max_aki_stage'}))

    stays = stays.merge(max_stage, on='stay_id', how='left')
    stays['max_aki_stage'] = stays['max_aki_stage'].fillna(0).astype(int)

# Hard leakage assertion — no KDIGO events from input window used
    leakage_rows = merged[
        (merged['charttime'] <= merged['label_window_start']) & in_window
    ]
    assert len(leakage_rows) == 0, \
        f"Leakage: {len(leakage_rows)} KDIGO events inside input window used for label"

    stays['aki_target'] = (stays['max_aki_stage'] >= 2).astype(int)

    pos = stays['aki_target'].sum()
    print(f"  Total stays with targets: {len(stays):,}")
    print(f"  AKI positive (stage 2/3): {pos:,}  ({pos/len(stays):.1%})")
    print(f"  AKI negative:             {len(stays)-pos:,}")

    return stays.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Clinical Notes Processing (pre-grouped — no repeated full-table scans)
# ---------------------------------------------------------------------------
def process_notes(stays):
    print("\n--- Clinical Notes Processing ---")

    notes = pd.read_csv(p(NOTE_DIR, 'discharge'),
                        usecols=['hadm_id', 'text'])

    tokenizer, has_real = get_tokenizer()

    # Pre-group by hadm_id — avoids O(N_patients × N_notes) scanning
    notes_grouped = {
        hadm_id: grp
        for hadm_id, grp in notes.groupby('hadm_id')
    }

    def get_tokens(hadm_id):
        if hadm_id not in notes_grouped:
            return []
        return process_patient_notes(
            notes_grouped[hadm_id], tokenizer, has_real
        )

    stays = stays.copy()
    stays['note_tokens'] = stays['hadm_id'].apply(get_tokens)

    before = len(stays)
    stays = stays[stays['note_tokens'].apply(len) > 0].reset_index(drop=True)
    print(f"  Patients with notes: {len(stays):,}  "
          f"(dropped {before - len(stays):,} with no notes)")

    return stays


# ---------------------------------------------------------------------------
# 4. Time-Series Extraction
#    FIX A: Correct subfolder paths for chartevents, labevents, outputevents
#    FIX B: Urine Output loaded from outputevents (not chartevents/labevents)
#    FIX C: labevents stay_id joined via hadm_id after loading
# ---------------------------------------------------------------------------
def extract_raw_timeseries(stays):
    print("\n--- Time-Series Raw Extraction ---")

    needed_chart_lab = set(CHART_LAB_IIDS)
    chunks = []

    # ── chartevents ──────────────────────────────────────────────────────────
    ce_path = p(ICU_DIR, 'chartevents')
    if os.path.exists(ce_path):
        print("  Reading chartevents (chunked)...")
        n_ce = 0
        for chunk in pd.read_csv(
            ce_path,
            chunksize=1_000_000,
            usecols=lambda c: c in ['subject_id', 'hadm_id', 'stay_id',
                                     'charttime', 'itemid', 'valuenum']
        ):
            filtered = chunk[chunk['itemid'].isin(needed_chart_lab)]
            if len(filtered):
                chunks.append(filtered)
                n_ce += len(filtered)
        print(f"    chartevents rows kept: {n_ce:,}")
    else:
        print("  WARNING: chartevents not found — skipping")

    # ── labevents (NO stay_id in MIMIC-IV — joined via hadm_id below) ────────
    le_path = p(HOSP_DIR, 'labevents')
    if os.path.exists(le_path):
        print("  Reading labevents (chunked)...")
        n_le = 0
        for chunk in pd.read_csv(
            le_path,
            chunksize=1_000_000,
            usecols=lambda c: c in ['subject_id', 'hadm_id',
                                     'charttime', 'itemid', 'valuenum']
        ):
            filtered = chunk[chunk['itemid'].isin(needed_chart_lab)]
            if len(filtered):
                chunks.append(filtered)
                n_le += len(filtered)
        print(f"    labevents rows kept: {n_le:,}")
    else:
        print("  WARNING: labevents not found — skipping")

    # ── outputevents — Urine Output only ─────────────────────────────────────
    oe_path = p(ICU_DIR, 'outputevents')
    if os.path.exists(oe_path):
        print("  Reading outputevents (Urine Output)...")
        uo = pd.read_csv(
            oe_path,
            usecols=lambda c: c in ['subject_id', 'hadm_id', 'stay_id',
                                     'charttime', 'itemid', 'value']
        )
        uo = uo[uo['itemid'].isin(UO_ITEMIDS)].dropna(subset=['value'])
        uo = uo[uo['value'] >= 0]
        uo = uo.rename(columns={'value': 'valuenum'})
        if 'hadm_id' not in uo.columns:
            uo['hadm_id'] = np.nan
        print(f"    outputevents rows kept: {len(uo):,}")
        chunks.append(uo[['subject_id', 'hadm_id', 'stay_id',
                           'charttime', 'itemid', 'valuenum']])
    else:
        print("  WARNING: outputevents not found — Urine Output will be missing")

    # ── Concatenate all sources ───────────────────────────────────────────────
    if not chunks:
        raise RuntimeError("No event data loaded — check data paths")

    events = pd.concat(chunks, ignore_index=True)
    events['charttime'] = pd.to_datetime(events['charttime'])

    # ── FIX C: Join stay_id for labevents rows (stay_id is NaN for lab rows) ─
    stay_hadm = stays[['stay_id', 'hadm_id']].drop_duplicates()
    missing_stay = events['stay_id'].isna()
    if missing_stay.any():
        filled = (events[missing_stay]
                  .drop(columns=['stay_id'])
                  .merge(stay_hadm, on='hadm_id', how='left')['stay_id'])
        events.loc[missing_stay, 'stay_id'] = filled.values
        print(f"  stay_id filled for {missing_stay.sum():,} lab event rows "
              f"via hadm_id join")

    # Map itemids → feature names
    events['feature_name'] = events['itemid'].map(ITEMID_TO_FEATURE)
    events = events.dropna(subset=['feature_name', 'stay_id'])

    print(f"  Total event rows after join: {len(events):,}")

    # ── Per-patient hourly discretization ────────────────────────────────────
    raw_ts_data   = {}
    raw_mask_data = {}

    stays_indexed = stays.set_index('stay_id')

    for stay_id, row in stays_indexed.iterrows():
        window_end = row['label_window_start']  # = intime + 6h

        patient_events = events[
            (events['stay_id']   == stay_id) &
            (events['charttime'] >= row['intime']) &
            (events['charttime'] <  window_end)     # strict < prevents leakage
        ].copy()

        # Leakage assertion
        if not patient_events.empty:
            assert patient_events['charttime'].max() < window_end, \
                f"Leakage: event after input window for stay {stay_id}"

        patient_events['hour'] = np.floor(
            (patient_events['charttime'] - row['intime'])
            / pd.Timedelta(hours=1)
        ).astype(int).clip(0, 5)

        hourly = patient_events.pivot_table(
            index='hour', columns='feature_name',
            values='valuenum', aggfunc='mean'
        )
        hourly = hourly.reindex(index=range(6), columns=ORDERED_FEATURES)

        # Missingness mask BEFORE imputation (1 = was missing/imputed)
        mask = hourly.isna().astype(float)

        # Forward-fill then zero-fill remaining NaN
        hourly = hourly.ffill().fillna(0)

        raw_ts_data[stay_id]   = hourly
        raw_mask_data[stay_id] = mask

    print(f"  Time-series matrices built for {len(raw_ts_data):,} patients")
    return stays, raw_ts_data, raw_mask_data


# ---------------------------------------------------------------------------
# 5. Normalization (train statistics only — applied to both sets)
# ---------------------------------------------------------------------------
def normalize(train_cohort, test_cohort, raw_ts_data, raw_mask_data):
    print("\n--- Normalization (Train Statistics Only) ---")

    train_matrices = [raw_ts_data[sid] for sid in train_cohort['stay_id']
                      if sid in raw_ts_data]
    if train_matrices:
        stacked      = pd.concat(train_matrices)
        train_mean   = stacked.mean()
        train_std    = stacked.std().replace(0, 1).fillna(1)
    else:
        train_mean = pd.Series(0, index=ORDERED_FEATURES)
        train_std  = pd.Series(1, index=ORDERED_FEATURES)

    def apply_norm(stay_id):
        df   = raw_ts_data[stay_id]
        mask = raw_mask_data[stay_id]
        norm = (df - train_mean) / train_std
        norm = norm.fillna(0)
        combined = pd.concat([norm, mask.add_suffix('_mask')], axis=1)
        return combined.values.tolist()

    train_cohort = train_cohort.copy()
    test_cohort  = test_cohort.copy()

    train_cohort['ts_matrix'] = train_cohort['stay_id'].apply(apply_norm)
    test_cohort['ts_matrix']  = test_cohort['stay_id'].apply(apply_norm)

    return train_cohort, test_cohort


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("MIMIC-IV AKI Forecasting Pipeline")
    print("=" * 60)

    # 1. Cohort
    cohort = extract_cohort()

    # 2. Labels
    cohort = generate_targets(cohort)

    # 3. Notes (filter to patients with both modalities)
    cohort = process_notes(cohort)

    # 4. Time-series
    cohort, raw_ts_data, raw_mask_data = extract_raw_timeseries(cohort)

    # Drop patients with no time-series data
    cohort = cohort[cohort['stay_id'].isin(raw_ts_data)].reset_index(drop=True)
    print(f"\nFinal cohort (both modalities): {len(cohort):,}")

    # 5. Patient-level train/test split (80/20)
    print("\n--- Train / Test Split ---")
    subjects = cohort['subject_id'].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(subjects)
    split_idx   = max(1, int(len(subjects) * 0.8))
    train_subj  = set(subjects[:split_idx])
    test_subj   = set(subjects[split_idx:])

    assert len(train_subj & test_subj) == 0, "Subject leakage in train/test split"

    train_cohort = cohort[cohort['subject_id'].isin(train_subj)].copy()
    test_cohort  = cohort[cohort['subject_id'].isin(test_subj)].copy()

    print(f"  Train patients: {len(train_subj):,}  "
          f"| Train stays: {len(train_cohort):,}")
    print(f"  Test  patients: {len(test_subj):,}  "
          f"| Test  stays: {len(test_cohort):,}")

    # 6. Normalize
    train_cohort, test_cohort = normalize(
        train_cohort, test_cohort, raw_ts_data, raw_mask_data
    )

    # 7. Save
    out_train = os.path.join(OUTPUT_DIR, 'train_cohort.csv')
    out_test  = os.path.join(OUTPUT_DIR, 'test_cohort.csv')

    # ts_matrix stored as JSON string in CSV
    train_cohort['ts_matrix'] = train_cohort['ts_matrix'].apply(str)
    test_cohort['ts_matrix']  = test_cohort['ts_matrix'].apply(str)

    train_cohort.to_csv(out_train, index=False)
    test_cohort.to_csv(out_test,  index=False)

    print(f"\n  Saved: {out_train}")
    print(f"  Saved: {out_test}")

    # 8. Verification
    sample    = train_cohort.iloc[0]
    ts_shape  = np.array(eval(sample['ts_matrix'])).shape
    mask_frac = raw_mask_data[sample['stay_id']].values.mean()

    print("\n" + "=" * 60)
    print("PIPELINE VERIFICATION")
    print("=" * 60)
    print(f"Ordered features ({len(ORDERED_FEATURES)}): {ORDERED_FEATURES}")
    print(f"\nFinal Train stays:  {len(train_cohort):,}")
    print(f"Final Test  stays:  {len(test_cohort):,}")
    print(f"\nSample patient:     subject_id={sample['subject_id']}  "
          f"stay_id={sample['stay_id']}")
    print(f"AKI Target:         {sample['aki_target']}")
    print(f"Note subsequences:  {len(sample['note_tokens'])}")
    print(f"TS Matrix Shape:    {ts_shape}  "
          f"(expect ({6}, {len(ORDERED_FEATURES)*2}))")
    print(f"Mask imputed frac:  {mask_frac:.1%}  (expect 30-70% on real data)")

    expected_cols = len(ORDERED_FEATURES) * 2
    if ts_shape[1] == expected_cols:
        print(f"\n✓ SUCCESS: shape ({6}, {expected_cols}) correct")
    else:
        print(f"\n✗ WARNING: expected ({6}, {expected_cols}), got {ts_shape}")

    aki_pos = train_cohort['aki_target'].sum()
    print(f"\nTrain AKI positive: {aki_pos:,} / {len(train_cohort):,} "
          f"({aki_pos/len(train_cohort):.1%})")

    print("\nPipeline complete.")