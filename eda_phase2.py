"""
Phase 2 EDA Script: Dataset Profiling, Match-Count Distribution, Noise Comparison & S1 Duplicates.

Profiles:
  1. SCHEMA & SHAPE:
     - Row counts, column names, dtypes for train_source1/2/3.tsv and train_ground_truth.tsv
     - Missingness (% null/NaN) per column per source
     - Country value counts in source1 (confirm US/India split, check for other values or typos)
  2. GROUND TRUTH MATCH-COUNT DISTRIBUTION:
     - Per-S1 match count distribution: % singletons (0 matches), % exactly 1, % 2-5, % 6+, mean, median, max
  3. PER-COUNTRY MATCH-COUNT BREAKDOWN:
     - Match-count distribution split by S1 country (US vs India)
  4. SOURCE NOISE PROFILE (sample-based):
     - Sample ~5,000 matched pairs for S1 <-> S2 and S1 <-> S3 separately
     - Normalized name-token Jaccard overlap & address-token Jaccard overlap distributions
     - Comparison of noise characteristics between Source 2 and Source 3
  5. DUPLICATE / NEAR-DUPLICATE CHECK IN S1:
     - Exact duplicate check on (business_name, business_address, country)
"""

import os
import sys
import time
import random
from collections import Counter
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "code", "business_entity_resolution", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from normalization import normalize_name, normalize_address
except ImportError:
    print("Warning: Could not import normalization.py from src. Using fallback tokenizers.")
    def normalize_name(s):
        return str(s).lower().strip() if pd.notna(s) else ""
    def normalize_address(s):
        return str(s).lower().strip() if pd.notna(s) else ""


def compute_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    inter = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return inter / union if union > 0 else 0.0


def profile_schema_and_missingness(file_path: str, chunksize: int = 500000) -> Tuple[int, Dict[str, str], Dict[str, int]]:
    """Stream a TSV file to calculate row count, dtypes, and missing values count without loading entirely into RAM."""
    total_rows = 0
    missing_counts = Counter()
    dtypes = {}

    first_chunk = True
    for chunk in pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""], chunksize=chunksize):
        if first_chunk:
            for col in chunk.columns:
                dtypes[col] = str(chunk[col].dtype)
            first_chunk = False

        total_rows += len(chunk)
        for col in chunk.columns:
            series = chunk[col]
            is_miss = series.isna() | (series == "") | (series.str.strip().str.lower() == "null") | (series.str.strip().str.lower() == "nan")
            missing_counts[col] += int(is_miss.sum())

    return total_rows, dtypes, dict(missing_counts)


def run_eda(dataset_dir: str = os.path.join(BASE_DIR, "dataset", "train")):
    t0 = time.time()
    print("=" * 90)
    print("PHASE 2 PARALLEL EDA: DATASET PROFILING & NOISE CHARACTERIZATION")
    print("=" * 90)
    print(f"Dataset Directory: {dataset_dir}\n")

    gt_path = os.path.join(dataset_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    s2_path = os.path.join(dataset_dir, "train_source2.tsv")
    s3_path = os.path.join(dataset_dir, "train_source3.tsv")

    # =========================================================================
    # PART 1: SCHEMA, SHAPE & MISSINGNESS
    # =========================================================================
    print("=" * 90)
    print("1. SCHEMA, SHAPE & MISSINGNESS PROFILING")
    print("=" * 90)

    files_to_profile = [
        ("train_ground_truth.tsv", gt_path),
        ("train_source1.tsv", s1_path),
        ("train_source2.tsv", s2_path),
        ("train_source3.tsv", s3_path),
    ]

    schema_summary = []
    for fname, fpath in files_to_profile:
        t_f0 = time.time()
        print(f"Profiling {fname}...")
        row_cnt, dtypes, miss_cnts = profile_schema_and_missingness(fpath)
        t_f1 = time.time()
        for col, dt in dtypes.items():
            miss_n = miss_cnts.get(col, 0)
            miss_pct = (miss_n / row_cnt * 100) if row_cnt > 0 else 0.0
            schema_summary.append({
                "File": fname,
                "Total Rows": row_cnt,
                "Column": col,
                "Dtype": dt,
                "Missing Count": miss_n,
                "Missing %": miss_pct,
                "Scan Time (s)": round(t_f1 - t_f0, 2)
            })

    schema_df = pd.DataFrame(schema_summary)
    print("\n--- Summary Schema & Missingness Table ---")
    hdr = f"{'File':<24} {'Total Rows':>12} {'Column':<18} {'Missing Count':>14} {'Missing %':>10}"
    print(hdr)
    print("-" * len(hdr))
    curr_f = None
    for _, r in schema_df.iterrows():
        f_display = r["File"] if r["File"] != curr_f else ""
        rows_display = f"{r['Total Rows']:,d}" if r["File"] != curr_f else ""
        print(f"{f_display:<24} {rows_display:>12} {r['Column']:<18} {r['Missing Count']:>14,d} {r['Missing %']:>9.4f}%")
        curr_f = r["File"]
    print("-" * len(hdr))

    # Load S1 in full for subsequent analyses
    print("\nLoading train_source1.tsv into memory for detailed checks...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    print(f"Loaded {len(s1_df):,} S1 rows.")

    # Country value counts in S1
    print("\n--- Source 1 Country Distribution & Value Check ---")
    c_counts = s1_df["country"].value_counts(dropna=False)
    for c_val, cnt in c_counts.items():
        pct = (cnt / len(s1_df)) * 100
        print(f"  Country '{c_val}': {cnt:,d} rows ({pct:.4f}%)")
    unique_countries = set(c_counts.index)
    if unique_countries == {"US", "India"}:
        print("  >> Verification: Exactly {'US', 'India'} present. No unexpected country values or typos.")
    else:
        print(f"  >> ALERT: Unexpected country values detected: {unique_countries - {'US', 'India'}}")

    # =========================================================================
    # PART 2: GROUND TRUTH MATCH-COUNT DISTRIBUTION
    # =========================================================================
    print("\n" + "=" * 90)
    print("2. GROUND TRUTH MATCH-COUNT DISTRIBUTION (ALL S1 ENTITIES)")
    print("=" * 90)

    print("Loading train_ground_truth.tsv...")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])

    def parse_matches(m_str):
        if not m_str or pd.isna(m_str):
            return []
        return [x.strip() for x in str(m_str).split(",") if x.strip()]

    gt_df["matches"] = gt_df["matched_entity_ids"].apply(parse_matches)
    gt_df["match_count"] = gt_df["matches"].apply(len)

    total_gt_s1 = len(gt_df)
    match_counts = gt_df["match_count"].values

    singletons = int(np.sum(match_counts == 0))
    exactly_1 = int(np.sum(match_counts == 1))
    c_2_to_5 = int(np.sum((match_counts >= 2) & (match_counts <= 5)))
    c_6_plus = int(np.sum(match_counts >= 6))

    mean_mc = float(np.mean(match_counts))
    median_mc = float(np.median(match_counts))
    p75_mc = float(np.percentile(match_counts, 75))
    p90_mc = float(np.percentile(match_counts, 90))
    p99_mc = float(np.percentile(match_counts, 99))
    max_mc = int(np.max(match_counts))
    total_true_pairs = int(np.sum(match_counts))

    print(f"  Total S1 Entities in GT      : {total_gt_s1:,d}")
    print(f"  Total True-Match Pairs       : {total_true_pairs:,d}")
    print(f"  Singletons (0 matches)       : {singletons:>10,d} ({singletons / total_gt_s1 * 100:>6.2f}%)")
    print(f"  Exactly 1 match              : {exactly_1:>10,d} ({exactly_1 / total_gt_s1 * 100:>6.2f}%)")
    print(f"  2 to 5 matches               : {c_2_to_5:>10,d} ({c_2_to_5 / total_gt_s1 * 100:>6.2f}%)")
    print(f"  6+ matches                   : {c_6_plus:>10,d} ({c_6_plus / total_gt_s1 * 100:>6.2f}%)")
    print("-" * 50)
    print(f"  Mean Matches per S1          : {mean_mc:.4f}")
    print(f"  Median (p50) Matches per S1  : {median_mc:.1f}")
    print(f"  75th Percentile (p75)        : {p75_mc:.1f}")
    print(f"  90th Percentile (p90)        : {p90_mc:.1f}")
    print(f"  99th Percentile (p99)        : {p99_mc:.1f}")
    print(f"  Max Matches for single S1    : {max_mc:,d}")

    # =========================================================================
    # PART 3: PER-COUNTRY MATCH-COUNT BREAKDOWN
    # =========================================================================
    print("\n" + "=" * 90)
    print("3. PER-COUNTRY MATCH-COUNT BREAKDOWN (US vs INDIA)")
    print("=" * 90)

    # Join S1 country into gt_df
    s1_country_map = dict(zip(s1_df["entity_id"], s1_df["country"]))
    gt_df["country"] = gt_df["source1_entity_id"].map(s1_country_map)

    c_hdr = f"{'Country':<10} {'S1 Count':>10} {'Singletons (0)':>16} {'Exactly 1':>14} {'2 to 5':>14} {'6+':>12} {'Mean':>8} {'p50':>6} {'p90':>6} {'Max':>6}"
    print(c_hdr)
    print("-" * len(c_hdr))

    for c in ["US", "India"]:
        sub = gt_df[gt_df["country"] == c]
        sub_n = len(sub)
        if sub_n == 0:
            continue
        sub_mc = sub["match_count"].values
        s_0 = int(np.sum(sub_mc == 0))
        s_1 = int(np.sum(sub_mc == 1))
        s_2_5 = int(np.sum((sub_mc >= 2) & (sub_mc <= 5)))
        s_6p = int(np.sum(sub_mc >= 6))
        s_mean = float(np.mean(sub_mc))
        s_p50 = float(np.median(sub_mc))
        s_p90 = float(np.percentile(sub_mc, 90))
        s_max = int(np.max(sub_mc))

        print(f"{c:<10} {sub_n:>10,d} {s_0:>8,d} ({s_0/sub_n*100:>4.2f}%) {s_1:>7,d} ({s_1/sub_n*100:>4.2f}%) {s_2_5:>7,d} ({s_2_5/sub_n*100:>4.2f}%) {s_6p:>6,d} ({s_6p/sub_n*100:>4.2f}%) {s_mean:>8.2f} {s_p50:>6.1f} {s_p90:>6.1f} {s_max:>6,d}")
    print("-" * len(c_hdr))

    # =========================================================================
    # PART 4: SOURCE NOISE PROFILE (S1 <-> S2 vs S1 <-> S3)
    # =========================================================================
    print("\n" + "=" * 90)
    print("4. SOURCE NOISE PROFILE (SAMPLE-BASED JACCARD OVERLAP: S1-S2 vs S1-S3)")
    print("=" * 90)

    # Collect pairs separated by target source prefix
    print("Collecting true match pairs by target source (S2 vs S3)...")
    s2_pairs = []
    s3_pairs = []

    for row in gt_df[["source1_entity_id", "matches"]].itertuples(index=False):
        s1_id = row.source1_entity_id
        for mid in row.matches:
            if mid.startswith("S2-"):
                s2_pairs.append((s1_id, mid))
            elif mid.startswith("S3-"):
                s3_pairs.append((s1_id, mid))

    print(f"Total True Match Pairs Available: S1-S2 = {len(s2_pairs):,d}, S1-S3 = {len(s3_pairs):,d}")

    SAMPLE_SIZE = 5000
    random.seed(42)
    sample_s2 = random.sample(s2_pairs, min(SAMPLE_SIZE, len(s2_pairs)))
    sample_s3 = random.sample(s3_pairs, min(SAMPLE_SIZE, len(s3_pairs)))

    sampled_s1_ids = {p[0] for p in sample_s2} | {p[0] for p in sample_s3}
    sampled_s2_ids = {p[1] for p in sample_s2}
    sampled_s3_ids = {p[1] for p in sample_s3}

    print(f"Sampled {len(sample_s2):,} S1-S2 pairs and {len(sample_s3):,} S1-S3 pairs.")
    print(f"Target entities to fetch: S2 = {len(sampled_s2_ids):,}, S3 = {len(sampled_s3_ids):,}")

    # Build S1 lookup dictionary for sampled S1
    s1_lookup = {}
    for row in s1_df[s1_df["entity_id"].isin(sampled_s1_ids)].itertuples(index=False):
        s1_lookup[row.entity_id] = (
            str(row.business_name) if pd.notna(row.business_name) else "",
            str(row.business_address) if pd.notna(row.business_address) else ""
        )

    # Stream S2 to retrieve sampled S2 entities
    print("Streaming train_source2.tsv to retrieve sampled S2 records...")
    s2_lookup = {}
    for chunk in pd.read_csv(s2_path, sep="\t", dtype=str,
                             usecols=["entity_id", "business_name", "business_address"],
                             keep_default_na=False, na_values=[""], chunksize=500000):
        sub_c = chunk[chunk["entity_id"].isin(sampled_s2_ids)]
        for row in sub_c.itertuples(index=False):
            s2_lookup[row.entity_id] = (
                str(row.business_name) if pd.notna(row.business_name) else "",
                str(row.business_address) if pd.notna(row.business_address) else ""
            )
        if len(s2_lookup) >= len(sampled_s2_ids):
            break

    # Stream S3 to retrieve sampled S3 entities
    print("Streaming train_source3.tsv to retrieve sampled S3 records...")
    s3_lookup = {}
    for chunk in pd.read_csv(s3_path, sep="\t", dtype=str,
                             usecols=["entity_id", "business_name", "business_address"],
                             keep_default_na=False, na_values=[""], chunksize=500000):
        sub_c = chunk[chunk["entity_id"].isin(sampled_s3_ids)]
        for row in sub_c.itertuples(index=False):
            s3_lookup[row.entity_id] = (
                str(row.business_name) if pd.notna(row.business_name) else "",
                str(row.business_address) if pd.notna(row.business_address) else ""
            )
        if len(s3_lookup) >= len(sampled_s3_ids):
            break

    print(f"Retrieved {len(s2_lookup):,} S2 entities and {len(s3_lookup):,} S3 entities.")

    def compute_pair_overlaps(pairs, target_lookup, desc):
        name_jaccards = []
        addr_jaccards = []

        for s1_id, tid in pairs:
            if s1_id not in s1_lookup or tid not in target_lookup:
                continue
            s1_name_raw, s1_addr_raw = s1_lookup[s1_id]
            t_name_raw, t_addr_raw = target_lookup[tid]

            s1_name_norm = normalize_name(s1_name_raw)
            t_name_norm = normalize_name(t_name_raw)
            n_tokens1 = set(s1_name_norm.split())
            n_tokens2 = set(t_name_norm.split())
            name_jaccards.append(compute_jaccard(n_tokens1, n_tokens2))

            s1_addr_norm = normalize_address(s1_addr_raw)
            t_addr_norm = normalize_address(t_addr_raw)
            a_tokens1 = set(s1_addr_norm.split())
            a_tokens2 = set(t_addr_norm.split())
            addr_jaccards.append(compute_jaccard(a_tokens1, a_tokens2))

        return np.array(name_jaccards), np.array(addr_jaccards)

    s2_n_jac, s2_a_jac = compute_pair_overlaps(sample_s2, s2_lookup, "S1-S2")
    s3_n_jac, s3_a_jac = compute_pair_overlaps(sample_s3, s3_lookup, "S1-S3")

    def print_distribution_table(arr_s2, arr_s3, metric_name):
        print(f"\n--- {metric_name} Distribution Comparison ---")
        cols = f"{'Source Pair':<14} {'Count':>8} {'Mean':>8} {'p25':>8} {'Median':>8} {'p75':>8} {'p90':>8} {'Zero Jac %':>12} {'Exact Jac (1.0) %':>18}"
        print(cols)
        print("-" * len(cols))
        for label, arr in [("S1 <-> S2", arr_s2), ("S1 <-> S3", arr_s3)]:
            n = len(arr)
            mean_v = float(np.mean(arr))
            p25_v = float(np.percentile(arr, 25))
            med_v = float(np.median(arr))
            p75_v = float(np.percentile(arr, 75))
            p90_v = float(np.percentile(arr, 90))
            zero_pct = float(np.sum(arr == 0.0) / n * 100)
            one_pct = float(np.sum(arr >= 0.9999) / n * 100)
            print(f"{label:<14} {n:>8,d} {mean_v:>8.4f} {p25_v:>8.4f} {med_v:>8.4f} {p75_v:>8.4f} {p90_v:>8.4f} {zero_pct:>11.2f}% {one_pct:>17.2f}%")
        print("-" * len(cols))

    print_distribution_table(s2_n_jac, s3_n_jac, "Normalized Business Name Token Jaccard")
    print_distribution_table(s2_a_jac, s3_a_jac, "Normalized Business Address Token Jaccard")

    print("\n  Summary Noise Comparison:")
    name_diff = float(np.mean(s3_n_jac) - np.mean(s2_n_jac))
    addr_diff = float(np.mean(s3_a_jac) - np.mean(s2_a_jac))
    print(f"    - Name Jaccard Mean Difference (S3 - S2)   : {name_diff:+.4f}")
    print(f"    - Address Jaccard Mean Difference (S3 - S2): {addr_diff:+.4f}")
    if abs(name_diff) < 0.03 and abs(addr_diff) < 0.03:
        print("    >> Observation: Noise levels are broadly comparable between S2 and S3.")
    else:
        higher_name = "S3" if name_diff > 0 else "S2"
        higher_addr = "S3" if addr_diff > 0 else "S2"
        print(f"    >> Observation: {higher_name} has higher name overlap; {higher_addr} has higher address overlap.")

    # =========================================================================
    # PART 5: DUPLICATE / NEAR-DUPLICATE CHECK IN S1
    # =========================================================================
    print("\n" + "=" * 90)
    print("5. DUPLICATE / NEAR-DUPLICATE CHECK IN SOURCE 1 (CLEAN REFERENCE TEST)")
    print("=" * 90)

    print("Checking for duplicate (business_name, business_address, country) tuples in S1...")
    dup_mask = s1_df.duplicated(subset=["business_name", "business_address", "country"], keep=False)
    dup_count = int(dup_mask.sum())

    print(f"  Total Rows in S1                  : {len(s1_df):,d}")
    print(f"  Unique S1 Entity IDs              : {s1_df['entity_id'].nunique():,d}")
    print(f"  Rows with Duplicate (Name, Addr, C): {dup_count:,d} ({dup_count / len(s1_df) * 100:.4f}%)")

    if dup_count == 0:
        print("  >> Verification: Source 1 has ZERO exact duplicate (name, address, country) rows.")
        print("     S1 is confirmed to be a clean, distinct reference table.")
    else:
        dup_groups = int(s1_df[dup_mask].groupby(["business_name", "business_address", "country"]).ngroups)
        print(f"  Distinct Duplicate Tuples         : {dup_groups:,d}")
        print(f"\n  >> WARNING: Found {dup_count:,d} rows belonging to {dup_groups:,d} duplicate groups in S1.")
        print("  Showing first 5 duplicate groups (up to 2 records per group):")
        dup_df = s1_df[dup_mask].sort_values(by=["business_name", "business_address", "country"])
        shown_groups = 0
        for (b_name, b_addr, b_ctry), grp in dup_df.groupby(["business_name", "business_address", "country"]):
            print(f"\n    Group {shown_groups + 1}: Name='{b_name}', Address='{b_addr}', Country='{b_ctry}'")
            for r in grp.head(2).itertuples(index=False):
                print(f"      Entity ID: {r.entity_id}")
            shown_groups += 1
            if shown_groups >= 5:
                break

    id_dups = s1_df["entity_id"].duplicated().sum()
    print(f"\n  Duplicate entity_id check in S1: {id_dups:,d} duplicates found.")

    total_time = time.time() - t0
    print("\n" + "=" * 90)
    print(f"PHASE 2 EDA COMPLETE (Runtime: {total_time:.1f}s / {total_time/60:.2f}min)")
    print("=" * 90)


if __name__ == "__main__":
    run_eda()
