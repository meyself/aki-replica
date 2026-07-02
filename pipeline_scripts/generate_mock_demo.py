import pandas as pd
import numpy as np
import os

os.makedirs('../data', exist_ok=True)

patients = pd.DataFrame({
    'subject_id': [10001, 10002, 10003, 10004],
    'gender': ['F', 'M', 'F', 'M'],
    'anchor_age': [25, 45, 17, 65],
    'anchor_year': [2100, 2105, 2110, 2115],
    'anchor_year_group': ['2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019'],
    'dod': [np.nan, np.nan, np.nan, np.nan]
})
patients.to_csv('../data/patients.csv', index=False)

admissions = pd.DataFrame({
    'subject_id': [10001, 10002, 10002, 10003, 10004],
    'hadm_id': [200001, 200002, 200003, 200004, 200005],
    'admittime': ['2100-01-01 10:00:00', '2105-05-05 10:00:00', '2106-06-06 10:00:00', '2110-10-10 10:00:00', '2115-11-11 10:00:00'],
    'dischtime': ['2100-01-10 10:00:00', '2105-05-15 10:00:00', '2106-06-16 10:00:00', '2110-10-20 10:00:00', '2115-11-21 10:00:00']
})
admissions.to_csv('../data/admissions.csv', index=False)

icustays = pd.DataFrame({
    'subject_id': [10001, 10002, 10002, 10003, 10004],
    'hadm_id': [200001, 200002, 200002, 200004, 200005], 
    'stay_id': [300001, 300002, 300003, 300004, 300005],
    'intime': ['2100-01-01 12:00:00', '2105-05-05 12:00:00', '2105-05-10 12:00:00', '2110-10-10 12:00:00', '2115-11-11 12:00:00'],
    'outtime': ['2100-01-05 12:00:00', '2105-05-08 12:00:00', '2105-05-12 12:00:00', '2110-10-15 12:00:00', '2115-11-15 12:00:00']
})
icustays.to_csv('../data/icustays.csv', index=False)

kdigo = pd.DataFrame({
    'subject_id': [10001, 10004, 10004],
    'hadm_id': [200001, 200005, 200005],
    'stay_id': [300001, 300005, 300005],
    'charttime': ['2100-01-01 20:00:00', '2115-11-11 15:00:00', '2115-11-11 22:00:00'],
    'aki_stage': [2, 1, 3] 
})
kdigo.to_csv('../data/kdigo_stages.csv', index=False)

discharge = pd.DataFrame({
    'note_id': ['1', '2', '3'],
    'subject_id': [10001, 10002, 10004],
    'hadm_id': [200001, 200002, 200005],
    'text': [
        "Chief Complaint: Patient has acute kidney issues.\nHistory of Present Illness: Patient is sick.\nPast Medical History: None.\nPhysical Exam: Normal.\n___ Name ___ was here.",
        "Chief Complaint: Fever.\nHistory of Present Illness: Has fever.\nPast Medical History: None.\nPhysical Exam: Normal.",
        "Chief Complaint: Weakness.\nHistory of Present Illness: Very weak.\nPast Medical History: Hypertension.\nPhysical Exam: Weak pulse. ___ Doctor ___ signed."
    ]
})
discharge.to_csv('../data/discharge.csv', index=False)

# Expanded chartevents for meaningful fraction
# We map several items so it isn't 100% missing. We'll map Temp C (223762), HR (220045), RR (220210)
chartevents = pd.DataFrame({
    'subject_id': [10001, 10001, 10004, 10004, 10004, 10004],
    'hadm_id': [200001, 200001, 200005, 200005, 200005, 200005],
    'stay_id': [300001, 300001, 300005, 300005, 300005, 300005],
    'charttime': [
        '2100-01-01 13:00:00', '2100-01-01 14:00:00', 
        '2115-11-11 13:00:00', '2115-11-11 14:00:00', '2115-11-11 14:30:00', '2115-11-11 17:00:00'
    ],
    'itemid': [220045, 220210, 220045, 223762, 220621, 220045], 
    'valuenum': [90, 20, 100, 37.5, 110, 95]
})
chartevents.to_csv('../data/chartevents.csv', index=False)

print("Mock data generated in ../data/")
