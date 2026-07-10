"""
generate_kdigo.py — KDIGO 2012 AKI staging via DuckDB

Uses DuckDB window functions to implement the same logic as the official
MIT-LCP MIMIC-IV kdigo_creatinine + kdigo_uo + kdigo_stages SQL concepts,
adapted for local CSV files (no BigQuery required).

Run:
    python generate_kdigo.py --data-root data

Output:
    data/derived/kdigo_stages.csv.gz
    data/derived/kdigo_stay_summary.csv.gz
"""

import argparse
import os
from pathlib import Path
import duckdb
import pandas as pd

def resolve(data_root: str, subdir: str, stem: str) -> str:
    gz  = os.path.join(data_root, subdir, stem + '.csv.gz')
    csv = os.path.join(data_root, subdir, stem + '.csv')
    if os.path.exists(gz):  return gz
    if os.path.exists(csv): return csv
    raise FileNotFoundError(f"Cannot find {gz} or {csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data', type=str)
    args   = parser.parse_args()
    DATA   = args.data_root
    OUT    = os.path.join(DATA, 'derived')
    os.makedirs(OUT, exist_ok=True)

    con = duckdb.connect()

    # ── Load MIMIC-IV tables as DuckDB views ─────────────────────────────────
    print("Loading MIMIC-IV tables into DuckDB...")
    views = {
        'icustays':     resolve(DATA, 'icu',  'icustays'),
        'admissions':   resolve(DATA, 'hosp', 'admissions'),
        'labevents':    resolve(DATA, 'hosp', 'labevents'),
        'outputevents': resolve(DATA, 'icu',  'outputevents'),
        'chartevents':  resolve(DATA, 'icu',  'chartevents'),
    }
    for name, path in views.items():
        print(f"  {name}: {path}")
        con.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{path}')"
        )

    # ── Step 1: kdigo_creatinine ──────────────────────────────────────────────
    # Mirrors MIT-LCP mimic-code kdigo_creatinine concept.
    # Dynamic rolling minima over strict prior windows (current value excluded).
    # Handles both 48h absolute rise and 7-day ratio criteria.
    print("\nComputing kdigo_creatinine...")
    con.execute("""
    CREATE TABLE kdigo_creatinine AS
    WITH cr AS (
        -- Pull creatinine from labevents (itemid 50912)
        -- Join to ICU stays via hadm_id (labevents has no stay_id)
        SELECT
            ie.stay_id,
            ie.intime,
            ie.outtime,
            le.charttime,
            le.valuenum AS creat
        FROM icustays ie
        INNER JOIN admissions adm
            ON ie.hadm_id = adm.hadm_id
        INNER JOIN labevents le
            ON le.hadm_id = adm.hadm_id
        WHERE le.itemid  = 50912
          AND le.valuenum > 0.1
          AND le.valuenum < 30.0
          -- Include 7 days before admission for baseline computation
          AND le.charttime >= ie.intime - INTERVAL '7 days'
          AND le.charttime <= ie.outtime
    ),
    cr_with_baseline AS (
        SELECT
            stay_id,
            intime,
            outtime,
            charttime,
            creat,
            -- Rolling min in STRICT prior 48h (excludes current row)
            MIN(creat) OVER (
                PARTITION BY stay_id
                ORDER BY charttime
                RANGE BETWEEN INTERVAL '48 hours' PRECEDING
                          AND INTERVAL '1 second'  PRECEDING
            ) AS creat_low_48h,
            -- Rolling min in STRICT prior 7 days (excludes current row)
            MIN(creat) OVER (
                PARTITION BY stay_id
                ORDER BY charttime
                RANGE BETWEEN INTERVAL '7 days'   PRECEDING
                          AND INTERVAL '1 second'  PRECEDING
            ) AS creat_low_7d
        FROM cr
    ),
    cr_staged AS (
        SELECT
            stay_id,
            charttime,
            creat,
            creat_low_48h,
            creat_low_7d,
            CASE
                -- Stage 3: 3x 7-day baseline OR >= 4.0 with acute rise
                WHEN creat_low_7d IS NOT NULL
                 AND creat >= creat_low_7d * 3.0
                    THEN 3
                WHEN creat >= 4.0
                 AND creat_low_48h IS NOT NULL
                 AND creat >= creat_low_48h + 0.3
                    THEN 3
                -- Stage 2: 2x 7-day baseline
                WHEN creat_low_7d IS NOT NULL
                 AND creat >= creat_low_7d * 2.0
                    THEN 2
                -- Stage 1: 1.5x 7-day baseline OR 0.3 rise in 48h
                WHEN creat_low_7d  IS NOT NULL
                 AND creat >= creat_low_7d * 1.5
                    THEN 1
                WHEN creat_low_48h IS NOT NULL
                 AND creat >= creat_low_48h + 0.3
                    THEN 1
                ELSE 0
            END AS aki_stage_creat
        FROM cr_with_baseline
        -- Only keep measurements from ICU admission onward
        WHERE charttime >= intime
    )
    SELECT * FROM cr_staged
    """)

    n_creat = con.execute("SELECT COUNT(*) FROM kdigo_creatinine").fetchone()[0]
    dist_c  = con.execute("""
        SELECT aki_stage_creat, COUNT(*) as n
        FROM kdigo_creatinine
        GROUP BY 1 ORDER BY 1
    """).fetchdf()
    print(f"  Creatinine events staged: {n_creat:,}")
    print("  Stage distribution:")
    for _, row in dist_c.iterrows():
        print(f"    Stage {int(row.aki_stage_creat)}: {int(row.n):>9,}  "
              f"({int(row.n)/n_creat:.1%})")

    # ── Step 2: kdigo_uo ──────────────────────────────────────────────────────
    # Mirrors MIT-LCP mimic-code kdigo_uo concept.
    # UO rate over 6h, 12h, 24h windows using documented elapsed time only.
    # Missing hours are NOT counted as zero output.
    print("\nComputing kdigo_uo...")
    con.execute("""
    CREATE TABLE kdigo_uo AS
    WITH uo_raw AS (
        -- Aggregate urine output per stay per timestamp
        SELECT
            ie.stay_id,
            ie.intime,
            ie.outtime,
            oe.charttime,
            SUM(oe.value) AS urineoutput
        FROM icustays ie
        INNER JOIN outputevents oe
            ON ie.stay_id = oe.stay_id
        WHERE oe.itemid IN (
            226559, 226560, 226561, 226584, 226563,
            226564, 226565, 226567, 226557, 226558
        )
          AND oe.value >= 0
          AND oe.charttime >= ie.intime
          AND oe.charttime <= ie.outtime
        GROUP BY ie.stay_id, ie.intime, ie.outtime, oe.charttime
    ),
    -- Get weight (first valid weight in ICU stay)
    wt AS (
        SELECT
            ie.stay_id,
            FIRST(ce.valuenum ORDER BY ce.charttime) AS weight_kg
        FROM icustays ie
        INNER JOIN chartevents ce
            ON ie.stay_id = ce.stay_id
        WHERE ce.itemid IN (226512, 224639)
          AND ce.valuenum BETWEEN 20 AND 300
          AND ce.charttime BETWEEN ie.intime AND ie.outtime
        GROUP BY ie.stay_id
    ),
    uo_with_weight AS (
        SELECT
            u.stay_id,
            u.intime,
            u.outtime,
            u.charttime,
            u.urineoutput,
            w.weight_kg,
            -- Windowed UO sums using documented events only
            SUM(u.urineoutput) OVER (
                PARTITION BY u.stay_id
                ORDER BY u.charttime
                RANGE BETWEEN INTERVAL '6  hours' PRECEDING AND CURRENT ROW
            ) AS urineoutput_6h,
            SUM(u.urineoutput) OVER (
                PARTITION BY u.stay_id
                ORDER BY u.charttime
                RANGE BETWEEN INTERVAL '12 hours' PRECEDING AND CURRENT ROW
            ) AS urineoutput_12h,
            SUM(u.urineoutput) OVER (
                PARTITION BY u.stay_id
                ORDER BY u.charttime
                RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
            ) AS urineoutput_24h,
            -- Documented time in each window (hours since first event in window)
            DATEDIFF('hour',
                MIN(u.charttime) OVER (
                    PARTITION BY u.stay_id
                    ORDER BY u.charttime
                    RANGE BETWEEN INTERVAL '6  hours' PRECEDING AND CURRENT ROW
                ), u.charttime) + 1 AS tm_6h,
            DATEDIFF('hour',
                MIN(u.charttime) OVER (
                    PARTITION BY u.stay_id
                    ORDER BY u.charttime
                    RANGE BETWEEN INTERVAL '12 hours' PRECEDING AND CURRENT ROW
                ), u.charttime) + 1 AS tm_12h,
            DATEDIFF('hour',
                MIN(u.charttime) OVER (
                    PARTITION BY u.stay_id
                    ORDER BY u.charttime
                    RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
                ), u.charttime) + 1 AS tm_24h
        FROM uo_raw u
        LEFT JOIN wt w ON u.stay_id = w.stay_id
    ),
    uo_rates AS (
        SELECT
            stay_id,
            charttime,
            weight_kg,
            urineoutput,
            -- UO rate (ml/kg/hr): only valid when window >= required duration
            CASE WHEN weight_kg > 0 AND tm_6h  >= 6
                 THEN urineoutput_6h  / weight_kg / tm_6h
                 ELSE NULL END AS uo_rt_6h,
            CASE WHEN weight_kg > 0 AND tm_12h >= 12
                 THEN urineoutput_12h / weight_kg / tm_12h
                 ELSE NULL END AS uo_rt_12h,
            CASE WHEN weight_kg > 0 AND tm_24h >= 24
                 THEN urineoutput_24h / weight_kg / tm_24h
                 ELSE NULL END AS uo_rt_24h,
            -- Anuria: zero UO over 12h documented window
            CASE WHEN tm_12h >= 12 AND urineoutput_12h = 0
                 THEN 1 ELSE 0 END AS anuria_12h
        FROM uo_with_weight
    )
    SELECT
        stay_id,
        charttime,
        weight_kg,
        uo_rt_6h,
        uo_rt_12h,
        uo_rt_24h,
        CASE
            -- Stage 3: < 0.3 ml/kg/hr over 24h OR anuria over 12h
            WHEN uo_rt_24h IS NOT NULL AND uo_rt_24h < 0.3  THEN 3
            WHEN anuria_12h = 1                              THEN 3
            -- Stage 2: < 0.5 ml/kg/hr over 12h
            WHEN uo_rt_12h IS NOT NULL AND uo_rt_12h < 0.5  THEN 2
            -- Stage 1: < 0.5 ml/kg/hr over 6h
            WHEN uo_rt_6h  IS NOT NULL AND uo_rt_6h  < 0.5  THEN 1
            -- Stage 0: evaluable but no criterion met
            WHEN uo_rt_6h IS NOT NULL                        THEN 0
            -- NULL: insufficient documented time to evaluate
            ELSE NULL
        END AS aki_stage_uo
    FROM uo_rates
    """)

    n_uo   = con.execute(
        "SELECT COUNT(*) FROM kdigo_uo WHERE aki_stage_uo IS NOT NULL"
    ).fetchone()[0]
    dist_u = con.execute("""
        SELECT aki_stage_uo, COUNT(*) as n
        FROM kdigo_uo
        WHERE aki_stage_uo IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).fetchdf()
    print(f"  Evaluable UO events: {n_uo:,}")
    print("  Stage distribution:")
    for _, row in dist_u.iterrows():
        print(f"    Stage {int(row.aki_stage_uo)}: {int(row.n):>9,}  "
              f"({int(row.n)/n_uo:.1%})")

    # ── Step 3: kdigo_stages — combine creatinine + UO ───────────────────────
    print("\nCombining creatinine and UO into kdigo_stages...")
    con.execute("""
    CREATE TABLE kdigo_stages AS
WITH combined AS (
        SELECT
            COALESCE(cr.stay_id,   uo.stay_id)   AS stay_id,
            COALESCE(cr.charttime, uo.charttime) AS charttime,
            cr.creat,
            cr.aki_stage_creat,
            uo.aki_stage_uo,
            GREATEST(
                COALESCE(cr.aki_stage_creat, 0),
                COALESCE(uo.aki_stage_uo,    0)
            ) AS aki_stage
        FROM kdigo_creatinine cr
        FULL OUTER JOIN kdigo_uo uo
            ON cr.stay_id  = uo.stay_id
           AND cr.charttime = uo.charttime
    )
    SELECT
        COALESCE(stay_id, stay_id) AS stay_id,
        COALESCE(charttime, charttime) AS charttime,
        aki_stage_creat,
        aki_stage_uo,
        aki_stage
    FROM combined
    ORDER BY stay_id, charttime
    """)

    # ── Step 4: Export ────────────────────────────────────────────────────────
    print("\nExporting kdigo_stages.csv.gz...")
    event_path   = os.path.join(OUT, 'kdigo_stages.csv.gz')
    summary_path = os.path.join(OUT, 'kdigo_stay_summary.csv.gz')

    kdigo = con.execute("SELECT * FROM kdigo_stages").fetchdf()
    kdigo['charttime'] = pd.to_datetime(kdigo['charttime'])

    # Deduplicate any remaining same-second duplicates (keep max stage)
    before = len(kdigo)
    kdigo = (kdigo.groupby(['stay_id', 'charttime'], as_index=False)
                  .agg({'aki_stage_creat': 'max',
                        'aki_stage_uo':    'max',
                        'aki_stage':       'max'}))
    print(f"  Rows after dedup: {len(kdigo):,}  "
          f"(removed {before - len(kdigo):,} same-second duplicates)")


    # Assert validity
    assert kdigo['aki_stage'].between(0, 3).all(), \
        "AKI stages outside 0-3 found"
    assert not kdigo.duplicated(['stay_id', 'charttime']).any(), \
        "Duplicates remain after deduplication — investigate"

    kdigo.to_csv(event_path, index=False, compression='gzip')
    print(f"  Saved: {event_path}  ({len(kdigo):,} rows)")

    # Stay-level summary
    print("\nBuilding stay-level summary...")
    icu_df = con.execute(
        "SELECT subject_id, hadm_id, stay_id, intime, outtime FROM icustays"
    ).fetchdf()

    stay_max = (kdigo.groupby('stay_id')['aki_stage']
                     .max()
                     .reset_index()
                     .rename(columns={'aki_stage': 'max_aki_stage'}))

    onset = (kdigo[kdigo['aki_stage'] >= 1]
             .groupby('stay_id')['charttime']
             .min()
             .reset_index()
             .rename(columns={'charttime': 'aki_onset_time'}))

    summary = (icu_df
               .merge(stay_max, on='stay_id', how='left')
               .merge(onset,    on='stay_id', how='left'))

    summary['has_aki']          = summary['max_aki_stage'] >= 1
    summary['has_stage_2_or_3'] = summary['max_aki_stage'] >= 2

    summary.to_csv(summary_path, index=False, compression='gzip')
    print(f"  Saved: {summary_path}  ({len(summary):,} rows)")

    # ── Final report ──────────────────────────────────────────────────────────
    total   = len(kdigo)
    dist    = kdigo['aki_stage'].value_counts().sort_index()
    ev_sum  = summary['max_aki_stage'].notna()

    print(f"\n{'='*55}")
    print("KDIGO STAGE DISTRIBUTION (event-level rows)")
    print(f"{'='*55}")
    for s, cnt in dist.items():
        print(f"  Stage {s}: {cnt:>9,}  ({cnt/total:.1%})")

    print(f"\n{'='*55}")
    print("KDIGO STAGE DISTRIBUTION (per ICU stay — max stage)")
    print(f"{'='*55}")
    stay_dist = summary[ev_sum]['max_aki_stage'].value_counts().sort_index()
    n_ev      = ev_sum.sum()
    for s, cnt in stay_dist.items():
        print(f"  Stage {s}: {cnt:>6,}  ({cnt/n_ev:.1%})")

    aki_rate    = summary.loc[ev_sum, 'max_aki_stage'].ge(1).mean()
    severe_rate = summary.loc[ev_sum, 'max_aki_stage'].ge(2).mean()
    print(f"\n  AKI Stage >=1 per evaluable stay: {aki_rate:.1%}")
    print(f"  AKI Stage >=2 per evaluable stay: {severe_rate:.1%}")
    print(f"\n  Note: paper's ~37% is AKI in the 6-18h PREDICTION WINDOW,")
    print(f"  not across the full stay. Run run_pipeline.py to see that rate.")
    print(f"\n✓ Done — proceed to: python run_pipeline.py")

if __name__ == '__main__':
    main()