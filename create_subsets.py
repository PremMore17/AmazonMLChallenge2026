#!/usr/bin/env python3
"""
create_subsets.py — Deterministic Small Subset Generation for Train & Validation.

Samples:
  - 2,000 S1 entity IDs from splits/train_s1_ids.txt
  - 500 S1 entity IDs from splits/val_s1_ids.txt
Preserves the joint (country x match_count_bucket) distribution:
  - country: US vs India
  - match count buckets: 0 (singleton), 1, 2-5, 6+
Deterministic with random seed = 42.

Writes:
  - splits/train_subset_2k_ids.txt
  - splits/val_subset_500_ids.txt
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_match_bucket(count: int) -> str:
    if count == 0:
        return "0"
    elif count == 1:
        return "1"
    elif 2 <= count <= 5:
        return "2-5"
    else:
        return "6+"


def create_subsets(
    train_ids_path: str = os.path.join(BASE_DIR, "splits", "train_s1_ids.txt"),
    val_ids_path: str = os.path.join(BASE_DIR, "splits", "val_s1_ids.txt"),
    s1_path: str = os.path.join(BASE_DIR, "dataset", "train", "train_source1.tsv"),
    gt_path: str = os.path.join(BASE_DIR, "dataset", "train", "train_ground_truth.tsv"),
    train_target_n: int = 2000,
    val_target_n: int = 500,
    seed: int = 42,
    output_dir: str = os.path.join(BASE_DIR, "splits")
) -> Tuple[List[str], List[str]]:
    t0 = time.time()
    print("=" * 80)
    print("STEP A: DETERMINISTIC STRATIFIED SMALL SUBSET SAMPLING (SEED=42)")
    print("=" * 80)

    # 1. Read Split IDs
    print(f"Loading split IDs from {train_ids_path} and {val_ids_path}...")
    with open(train_ids_path, "r", encoding="utf-8") as f:
        train_ids = [line.strip() for line in f if line.strip()]
    with open(val_ids_path, "r", encoding="utf-8") as f:
        val_ids = [line.strip() for line in f if line.strip()]

    print(f"  Total train split S1 IDs: {len(train_ids):,}")
    print(f"  Total val split S1 IDs  : {len(val_ids):,}")

    train_set = set(train_ids)
    val_set = set(val_ids)
    all_needed_ids = train_set | val_set

    # 2. Load Country Mapping from train_source1.tsv
    print(f"\nStreaming S1 country metadata from {s1_path}...")
    s1_country_map: Dict[str, str] = {}
    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        usecols=["entity_id", "country"],
        keep_default_na=False,
        na_values=[""],
        chunksize=500000,
        low_memory=False
    ):
        for row in chunk.itertuples(index=False):
            eid = str(row.entity_id)
            if eid in all_needed_ids:
                c = str(row.country).strip()
                s1_country_map[eid] = c if c else "Unknown"

    print(f"  Found country metadata for {len(s1_country_map):,} entities.")

    # 3. Load Ground Truth Match Counts from train_ground_truth.tsv
    print(f"\nStreaming match counts from {gt_path}...")
    s1_match_counts: Dict[str, int] = defaultdict(int)
    for chunk in pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        usecols=["source1_entity_id", "matched_entity_ids"],
        keep_default_na=False,
        na_values=[""],
        chunksize=500000,
        low_memory=False
    ):
        for row in chunk.itertuples(index=False):
            s1_id = str(row.source1_entity_id)
            if s1_id in all_needed_ids:
                raw_m = str(row.matched_entity_ids) if pd.notna(row.matched_entity_ids) else ""
                ids = [x.strip() for x in raw_m.split(",") if x.strip()]
                s1_match_counts[s1_id] = len(ids)

    print(f"  Found ground truth entries for {len(s1_match_counts):,} entities (unlisted are singletons).")

    # 4. Form Strata for Train and Val
    # Stratum key: (country, match_bucket)
    def stratify_ids(id_list: List[str]) -> Dict[Tuple[str, str], List[str]]:
        strata = defaultdict(list)
        for eid in id_list:
            country = s1_country_map.get(eid, "Unknown")
            mc = s1_match_counts.get(eid, 0)
            bucket = get_match_bucket(mc)
            strata[(country, bucket)].append(eid)
        return strata

    train_strata = stratify_ids(train_ids)
    val_strata = stratify_ids(val_ids)

    # 5. Deterministic Proportional Sampling
    def sample_stratified(
        strata: Dict[Tuple[str, str], List[str]],
        target_n: int,
        seed_offset: int
    ) -> List[str]:
        total_available = sum(len(v) for v in strata.values())
        sampled_all = []

        # Sort strata keys for reproducibility
        sorted_keys = sorted(list(strata.keys()))
        rng = random.Random(seed + seed_offset)

        # First pass: proportional allocation
        allocated = {}
        for k in sorted_keys:
            items = strata[k]
            prop = len(items) / total_available
            n_sample = int(round(prop * target_n))
            # At least 1 if group is non-empty and target allows
            if n_sample == 0 and len(items) > 0 and sum(allocated.values()) < target_n:
                n_sample = 1
            allocated[k] = min(n_sample, len(items))

        # Adjust total allocation to match target_n exactly
        current_total = sum(allocated.values())
        diff = target_n - current_total

        if diff > 0:
            # Add remaining to largest groups
            sort_by_size = sorted(sorted_keys, key=lambda k: len(strata[k]), reverse=True)
            for k in sort_by_size:
                if diff == 0:
                    break
                can_add = len(strata[k]) - allocated[k]
                add = min(can_add, diff)
                allocated[k] += add
                diff -= add
        elif diff < 0:
            # Remove from largest allocated groups
            sort_by_alloc = sorted(sorted_keys, key=lambda k: allocated[k], reverse=True)
            for k in sort_by_alloc:
                if diff == 0:
                    break
                can_sub = max(0, allocated[k] - 1)
                sub = min(can_sub, -diff)
                allocated[k] -= sub
                diff += sub

        # Sample deterministically within each stratum
        for k in sorted_keys:
            items = sorted(strata[k])  # Sort for stable RNG shuffle
            rng.shuffle(items)
            sampled_all.extend(items[:allocated[k]])

        rng.shuffle(sampled_all)
        return sampled_all

    print(f"\nSampling {train_target_n:,} train IDs and {val_target_n:,} val IDs...")
    sampled_train_ids = sample_stratified(train_strata, train_target_n, seed_offset=0)
    sampled_val_ids = sample_stratified(val_strata, val_target_n, seed_offset=100)

    # 6. Verify and Report Distribution
    def print_strata_report(title: str, full_strata: Dict, sampled_ids: List[str]):
        print(f"\n--- {title} Distribution Comparison ---")
        sampled_strata = defaultdict(int)
        for eid in sampled_ids:
            c = s1_country_map.get(eid, "Unknown")
            b = get_match_bucket(s1_match_counts.get(eid, 0))
            sampled_strata[(c, b)] += 1

        full_total = sum(len(v) for v in full_strata.values())
        samp_total = len(sampled_ids)

        print(f"{'Country':<10} {'Bucket':<8} {'Full Count':>12} {'Full %':>10} {'Sample Count':>14} {'Sample %':>10}")
        print("-" * 70)
        for k in sorted(full_strata.keys()):
            f_cnt = len(full_strata[k])
            f_pct = f_cnt / full_total * 100
            s_cnt = sampled_strata.get(k, 0)
            s_pct = s_cnt / samp_total * 100 if samp_total else 0
            print(f"{k[0]:<10} {k[1]:<8} {f_cnt:>12,d} {f_pct:>9.2f}% {s_cnt:>14,d} {s_pct:>9.2f}%")
        print("-" * 70)

    print_strata_report("Train Set (2,000 Subset)", train_strata, sampled_train_ids)
    print_strata_report("Validation Set (500 Subset)", val_strata, sampled_val_ids)

    # 7. Write Output Files
    os.makedirs(output_dir, exist_ok=True)
    out_train_path = os.path.join(output_dir, "train_subset_2k_ids.txt")
    out_val_path = os.path.join(output_dir, "val_subset_500_ids.txt")

    print(f"\nWriting {len(sampled_train_ids):,} IDs to {out_train_path}...")
    with open(out_train_path, "w", encoding="utf-8") as f:
        for eid in sampled_train_ids:
            f.write(f"{eid}\n")

    print(f"Writing {len(sampled_val_ids):,} IDs to {out_val_path}...")
    with open(out_val_path, "w", encoding="utf-8") as f:
        for eid in sampled_val_ids:
            f.write(f"{eid}\n")

    elapsed = time.time() - t0
    print(f"\nSTEP A COMPLETE in {elapsed:.1f}s.")
    return sampled_train_ids, sampled_val_ids


if __name__ == "__main__":
    create_subsets()
