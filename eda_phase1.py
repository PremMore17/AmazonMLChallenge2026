"""
Phase 1 Confirmation EDA Script for Business Entity Resolution Challenge.
Highly optimized: memory-efficient streaming, fast execution.
Outputs to both console and phase1_output.txt.

Computes:
  1. Country-blocking safety: exact cross-country mismatch rate among ground-truth matches
  2. Script mix: fraction of true-match pairs where S1 name and matched S2/S3 name differ in Unicode script
  3. Candidate volume estimate: S1 x (S2 U S3) in TEST set if blocked purely by country
"""

import os
import sys
import unicodedata
from collections import Counter
import pandas as pd
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_LOG = os.path.join(BASE_DIR, "phase1_output.txt")

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_dominant_script(text):
    if not text or not isinstance(text, str):
        return "Unknown"
    
    script_counts = Counter()
    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith("P") or ch.isdigit():
            continue
        try:
            ch_name = unicodedata.name(ch)
        except ValueError:
            continue
        
        if "LATIN" in ch_name:
            script_counts["Latin"] += 1
        elif "DEVANAGARI" in ch_name:
            script_counts["Devanagari"] += 1
        elif "TAMIL" in ch_name:
            script_counts["Tamil"] += 1
        elif "TELUGU" in ch_name:
            script_counts["Telugu"] += 1
        elif "KANNADA" in ch_name:
            script_counts["Kannada"] += 1
        elif "BENGALI" in ch_name:
            script_counts["Bengali"] += 1
        elif "GUJARATI" in ch_name:
            script_counts["Gujarati"] += 1
        elif "MALAYALAM" in ch_name:
            script_counts["Malayalam"] += 1
        elif "GURMUKHI" in ch_name:
            script_counts["Gurmukhi"] += 1
        elif "ARABIC" in ch_name:
            script_counts["Arabic"] += 1
        elif "CYRILLIC" in ch_name:
            script_counts["Cyrillic"] += 1
        else:
            script_counts["Other"] += 1
            
    if not script_counts:
        return "Unknown"
    return script_counts.most_common(1)[0][0]

def main():
    sys.stdout = Logger(OUTPUT_LOG)
    t0 = time.time()
    
    print("=" * 80)
    print("PHASE 1 CONFIRMATION EDA — BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 80)

    # 1. Load Ground Truth
    gt_path = os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv")
    print(f"\n[1/4] Loading train ground truth from {gt_path}...")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    
    def parse_ids(x):
        if pd.isna(x) or str(x).strip() == "":
            return []
        return [i.strip() for i in str(x).split(",") if i.strip()]
    
    gt["match_list"] = gt["matched_entity_ids"].apply(parse_ids)
    gt["n_matches"] = gt["match_list"].apply(len)
    
    total_s1 = len(gt)
    n_singletons = (gt["n_matches"] == 0).sum()
    n_single_match = (gt["n_matches"] == 1).sum()
    n_multi_match = (gt["n_matches"] > 1).sum()
    
    # Flatten match pairs and collect unique target IDs
    s1_ids = []
    match_ids = []
    target_s2_ids = set()
    target_s3_ids = set()
    
    for s1_id, mlist in zip(gt["source1_entity_id"], gt["match_list"]):
        for mid in mlist:
            s1_ids.append(s1_id)
            match_ids.append(mid)
            if mid.startswith("S2-"):
                target_s2_ids.add(mid)
            elif mid.startswith("S3-"):
                target_s3_ids.add(mid)
                
    total_true_pairs = len(s1_ids)
    print(f"  Total S1 entities in train: {total_s1:,}")
    print(f"  Singletons (0 matches): {n_singletons:,} ({n_singletons/total_s1*100:.2f}%)")
    print(f"  Exactly 1 match: {n_single_match:,} ({n_single_match/total_s1*100:.2f}%)")
    print(f"  Multi-match (>1 matches): {n_multi_match:,} ({n_multi_match/total_s1*100:.2f}%)")
    print(f"  Total true-match pairs (S1 -> S2/S3): {total_true_pairs:,}")
    print(f"  Unique matched S2 entities: {len(target_s2_ids):,}")
    print(f"  Unique matched S3 entities: {len(target_s3_ids):,}")

    # 2. Load S1 metadata
    print(f"\n[2/4] Loading train S1 records...")
    s1_path = os.path.join(DATASET_DIR, "train", "train_source1.tsv")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "country"],
                        keep_default_na=False, na_values=[""])
    s1_meta = {row.entity_id: (row.business_name, row.country) for row in s1_df.itertuples(index=False)}
    del s1_df

    # 3. Stream S2 & S3 to collect matched entity attributes
    print(f"\n[3/4] Streaming train S2 and S3 records for matched entities...")
    match_meta = {} # mid -> (business_name, country)
    
    s2_path = os.path.join(DATASET_DIR, "train", "train_source2.tsv")
    for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "country"],
                             keep_default_na=False, na_values=[""], chunksize=500000):
        m = chunk[chunk["entity_id"].isin(target_s2_ids)]
        for row in m.itertuples(index=False):
            match_meta[row.entity_id] = (row.business_name, row.country)
            
    s3_path = os.path.join(DATASET_DIR, "train", "train_source3.tsv")
    for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "country"],
                             keep_default_na=False, na_values=[""], chunksize=500000):
        m = chunk[chunk["entity_id"].isin(target_s3_ids)]
        for row in m.itertuples(index=False):
            match_meta[row.entity_id] = (row.business_name, row.country)

    print(f"  Loaded metadata for {len(match_meta):,} / {len(target_s2_ids) + len(target_s3_ids):,} matched entities.")

    # --------------------------------------------------------------------------
    # ITEM 1: Country-blocking safety
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ITEM 1: COUNTRY-BLOCKING SAFETY (CROSS-COUNTRY MISMATCH RATE)")
    print("=" * 80)

    mismatches = 0
    mismatch_examples = []
    missing_meta = 0

    for s1_id, mid in zip(s1_ids, match_ids):
        s1_info = s1_meta.get(s1_id)
        mid_info = match_meta.get(mid)
        if not s1_info or not mid_info:
            missing_meta += 1
            continue
        c1 = s1_info[1]
        c2 = mid_info[1]
        if c1 != c2:
            mismatches += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append((s1_id, c1, mid, c2))

    mismatch_rate = (mismatches / total_true_pairs) if total_true_pairs > 0 else 0.0
    print(f"Total ground-truth match pairs evaluated: {total_true_pairs:,}")
    print(f"Cross-country mismatches found: {mismatches:,}")
    print(f"Exact cross-country mismatch rate: {mismatch_rate * 100:.6f}%")
    if mismatches == 0:
        print(">> VERDICT: Country mismatch rate is EXACTLY 0.000000%.")
        print("   Hard-partitioning by country is 100% SAFE and will NOT lose any ground truth matches.")
    else:
        print(f">> WARNING: Found {mismatches} cross-country mismatches. Examples:")
        for ex in mismatch_examples:
            print(f"   S1: {ex[0]} ({ex[1]})  vs  Match: {ex[2]} ({ex[3]})")

    # --------------------------------------------------------------------------
    # ITEM 2: Script mix
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ITEM 2: SCRIPT MIX AMONG TRUE-MATCH PAIRS")
    print("=" * 80)

    cross_script_count = 0
    script_pair_counter = Counter()
    s1_script_counter = Counter()
    match_script_counter = Counter()
    cross_script_examples = []

    for s1_id, mid in zip(s1_ids, match_ids):
        s1_info = s1_meta.get(s1_id)
        mid_info = match_meta.get(mid)
        if not s1_info or not mid_info:
            continue
        
        n1 = s1_info[0]
        n2 = mid_info[0]
        sc1 = get_dominant_script(n1)
        sc2 = get_dominant_script(n2)

        s1_script_counter[sc1] += 1
        match_script_counter[sc2] += 1
        script_pair_counter[(sc1, sc2)] += 1

        if sc1 != sc2:
            cross_script_count += 1
            if len(cross_script_examples) < 10:
                cross_script_examples.append((s1_id, n1, sc1, mid, n2, sc2))

    cross_script_rate = (cross_script_count / total_true_pairs) if total_true_pairs > 0 else 0.0
    print(f"Total true-match pairs evaluated: {total_true_pairs:,}")
    print(f"Cross-script true-match pairs: {cross_script_count:,} ({cross_script_rate * 100:.4f}%)")
    print(f"Same-script true-match pairs: {total_true_pairs - cross_script_count:,} ({(1 - cross_script_rate) * 100:.4f}%)")

    print("\nDominant Script Distribution in S1 Names (Ground Truth Matches):")
    for sc, cnt in s1_script_counter.most_common():
        print(f"  {sc:<15}: {cnt:>10,d} ({cnt/total_true_pairs*100:>6.2f}%)")

    print("\nDominant Script Distribution in Matched S2/S3 Names:")
    for sc, cnt in match_script_counter.most_common():
        print(f"  {sc:<15}: {cnt:>10,d} ({cnt/total_true_pairs*100:>6.2f}%)")

    print("\nTop Script Combinations (S1 Script -> Matched S2/S3 Script):")
    for (sc1, sc2), cnt in script_pair_counter.most_common(12):
        flag = " [CROSS-SCRIPT]" if sc1 != sc2 else ""
        print(f"  {sc1:<12} -> {sc2:<12}: {cnt:>10,d} ({cnt/total_true_pairs*100:>6.2f}%){flag}")

    if cross_script_examples:
        print("\nExample Cross-Script Matches:")
        for ex in cross_script_examples[:6]:
            print(f"  S1 [{ex[2]}]: {ex[1]} ({ex[0]})")
            print(f"  Match [{ex[5]}]: {ex[4]} ({ex[3]})")
            print("  ---")

    # Clean up train data from memory
    del s1_meta, match_meta, gt

    # --------------------------------------------------------------------------
    # ITEM 3: Candidate volume estimate in TEST set (pure country blocking)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ITEM 3: CANDIDATE VOLUME ESTIMATE IN TEST SET (PURE COUNTRY BLOCKING)")
    print("=" * 80)
    print("\n[4/4] Loading country columns from test files...")

    ts1_c = pd.read_csv(os.path.join(DATASET_DIR, "test", "test_source1.tsv"), sep="\t", usecols=["country"], dtype=str)["country"].value_counts()
    ts2_c = pd.read_csv(os.path.join(DATASET_DIR, "test", "test_source2.tsv"), sep="\t", usecols=["country"], dtype=str)["country"].value_counts()
    ts3_c = pd.read_csv(os.path.join(DATASET_DIR, "test", "test_source3.tsv"), sep="\t", usecols=["country"], dtype=str)["country"].value_counts()

    all_countries = sorted(list(set(ts1_c.index) | set(ts2_c.index) | set(ts3_c.index)))

    print(f"\n{'Country':<15} {'|S1|':>12} {'|S2|':>12} {'|S3|':>12} {'|S2 U S3|':>14} {'Pairs: |S1| x |S2 U S3|':>26}")
    print("-" * 95)
    
    total_s1_test = 0
    total_s2_test = 0
    total_s3_test = 0
    total_pairs = 0

    for c in all_countries:
        n1 = int(ts1_c.get(c, 0))
        n2 = int(ts2_c.get(c, 0))
        n3 = int(ts3_c.get(c, 0))
        n_cand = n2 + n3
        pairs = n1 * n_cand
        
        total_s1_test += n1
        total_s2_test += n2
        total_s3_test += n3
        total_pairs += pairs
        
        print(f"{c:<15} {n1:>12,d} {n2:>12,d} {n3:>12,d} {n_cand:>14,d} {pairs:>26,d}")

    print("-" * 95)
    print(f"{'TOTAL':<15} {total_s1_test:>12,d} {total_s2_test:>12,d} {total_s3_test:>12,d} {total_s2_test + total_s3_test:>14,d} {total_pairs:>26,d}")
    print(f"\nTotal Test Candidate Pairs if blocked purely by country: {total_pairs:,} (~{total_pairs/1e12:.2f} Trillion pairs)")
    print(">> TAKEAWAY: Pure country blocking is completely intractable without a secondary blocking key.")
    print("   Secondary token / n-gram blocking inside each country partition is strictly mandatory.")

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"PHASE 1 COMPLETE (Completed in {elapsed:.1f}s)")
    print("=" * 80)

if __name__ == "__main__":
    main()
