"""
Phase 3 Split Script: Entity-Level Stratified Train/Validation Split.

Splits Source 1 entities and ground truth into train and validation sets,
stratifying by ground-truth match-count bucket (singleton, exactly-1, 2-5, 6+)
to preserve rare-case ratios and prevent cross-source entity leakage.

Outputs:
  - train_s1_ids.txt
  - val_s1_ids.txt
  - train_gt_rows.tsv
  - val_gt_rows.tsv
  - split_stats.txt
"""

import os
import sys
import time
import random
import argparse
from collections import Counter
from typing import List, Dict, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Entity-Level Stratified Train/Val Split")
    parser.add_argument("--s1", type=str, default="dataset/train/train_source1.tsv",
                        help="Path to train_source1.tsv")
    parser.add_argument("--gt", type=str, default="dataset/train/train_ground_truth.tsv",
                        help="Path to train_ground_truth.tsv")
    parser.add_argument("--out_dir", type=str, default="splits",
                        help="Output directory to save splits (default: 'splits/')")
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="Fraction of S1 entities to assign to validation set (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def load_tsv(path: str) -> pd.DataFrame:
    """Robust TSV loading matching eda_phase2.py pattern."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])


def assign_bucket(match_count: int) -> str:
    """Assigns an S1 entity to a match-count stratum bucket."""
    if match_count == 0:
        return "singleton"
    elif match_count == 1:
        return "exactly-1"
    elif 2 <= match_count <= 5:
        return "2-5"
    else:
        return "6+"


def perform_split(
    s1_ids: List[str],
    bucket_labels: List[str],
    val_frac: float,
    seed: int
) -> Tuple[List[str], List[str]]:
    """
    Performs stratified train/test split.
    Converts inputs to plain Python lists before passing to sklearn train_test_split
    to avoid PyArrow ChunkedArray / ArrowExtensionArray fancy indexing incompatibilities.
    If any class has < 2 members, falls back to manual per-stratum split.
    """
    s1_ids_list = list(s1_ids)
    bucket_labels_list = list(bucket_labels)

    counts = Counter(bucket_labels_list)
    min_count = min(counts.values()) if counts else 0

    if min_count >= 2:
        train_ids, val_ids = train_test_split(
            s1_ids_list,
            test_size=val_frac,
            random_state=seed,
            stratify=bucket_labels_list
        )
        return list(train_ids), list(val_ids)
    else:
        print("  [Notice] Stratum with < 2 members detected. Falling back to manual per-stratum split.")
        rng = random.Random(seed)
        train_list = []
        val_list = []
        for bucket in sorted(counts.keys()):
            bucket_ids = [s1_ids_list[i] for i in range(len(s1_ids_list)) if bucket_labels_list[i] == bucket]
            rng.shuffle(bucket_ids)
            n_bucket = len(bucket_ids)
            n_val = max(1 if n_bucket > 1 and val_frac > 0 else 0, int(round(n_bucket * val_frac)))
            val_list.extend(bucket_ids[:n_val])
            train_list.extend(bucket_ids[n_val:])
        return train_list, val_list


def main():
    t0 = time.time()
    args = parse_args()

    print("=" * 80)
    print("PHASE 3: ENTITY-LEVEL STRATIFIED TRAIN/VAL SPLIT")
    print("=" * 80)
    print(f"Source 1 Path      : {args.s1}")
    print(f"Ground Truth Path  : {args.gt}")
    print(f"Output Directory   : {args.out_dir}")
    print(f"Validation Fraction: {args.val_frac:.4f} ({args.val_frac * 100:.1f}%)")
    print(f"Random Seed        : {args.seed}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load data
    print("Loading train_source1.tsv...")
    s1_df = load_tsv(args.s1)
    # Convert s1_ids immediately to a plain Python list of strings
    # This prevents any downstream ArrowExtensionArray / PyArrow indexing errors
    s1_ids: List[str] = [str(x) for x in s1_df["entity_id"].tolist()]
    total_s1 = len(s1_ids)
    print(f"  Loaded {total_s1:,d} S1 entities.")

    print("\nLoading train_ground_truth.tsv...")
    gt_df = load_tsv(args.gt)
    print(f"  Loaded {len(gt_df):,} ground truth rows.")

    # 2. Parse matches and compute match_count using native Python collections
    print("\nParsing match counts per S1 entity...")
    gt_s1_col = [str(x) for x in gt_df["source1_entity_id"].tolist()]
    gt_m_col = [str(x) for x in gt_df["matched_entity_ids"].tolist()]

    gt_match_dict: Dict[str, int] = {}
    for sid, m_str in zip(gt_s1_col, gt_m_col):
        if not m_str or m_str.strip() == "" or m_str.lower() in ("nan", "null"):
            gt_match_dict[sid] = 0
        else:
            tokens = [x for x in m_str.split(",") if x.strip()]
            gt_match_dict[sid] = len(tokens)

    # Map match count onto all S1 IDs as a plain Python list
    match_counts: List[int] = [gt_match_dict.get(sid, 0) for sid in s1_ids]
    bucket_labels: List[str] = [assign_bucket(mc) for mc in match_counts]

    bucket_order = ["singleton", "exactly-1", "2-5", "6+"]
    print("\nOverall Ground Truth Strata Breakdown:")
    total_counts = Counter(bucket_labels)
    for b in bucket_order:
        cnt = total_counts.get(b, 0)
        pct = (cnt / total_s1 * 100) if total_s1 > 0 else 0.0
        print(f"  - {b:<12}: {cnt:>10,d} ({pct:>6.2f}%)")

    # 3. Perform stratified split with plain lists
    print(f"\nPerforming stratified split (val_frac={args.val_frac}, seed={args.seed})...")
    train_ids, val_ids = perform_split(s1_ids, bucket_labels, args.val_frac, args.seed)

    # Convert to sets for fast membership testing
    train_s1_set = set(train_ids)
    val_s1_set = set(val_ids)

    # 4. Sanity checks
    print("\nRunning Sanity Checks:")
    overlap = train_s1_set & val_s1_set
    if len(overlap) == 0:
        print(f"  [1] Train/Val S1 ID overlap       : 0 (PASSED: zero overlap)")
    else:
        print(f"  [FAIL] Overlap count: {len(overlap)}")
    sum_check = (len(train_ids) + len(val_ids) == total_s1)
    if sum_check:
        print(f"  [2] Length conservation check     : {len(train_ids):,d} + {len(val_ids):,d} = {len(train_ids)+len(val_ids):,d} == {total_s1:,d} (PASSED)")
    else:
        print(f"  [FAIL] Length mismatch: {len(train_ids)+len(val_ids)} vs {total_s1}")
    assert len(overlap) == 0, "Train and Val S1 entity ID sets must not overlap!"
    assert sum_check, "Train and Val S1 entity ID lengths must sum to total S1 count!"

    # 5. Build and print verification table
    id_to_bucket = dict(zip(s1_ids, bucket_labels))
    train_buckets = Counter(id_to_bucket[sid] for sid in train_ids)
    val_buckets = Counter(id_to_bucket[sid] for sid in val_ids)

    table_lines = []
    table_lines.append("=" * 88)
    table_lines.append(f"{'Stratum Bucket':<16} {'Total N':>12} {'Total %':>10} {'Train N':>12} {'Train %':>10} {'Val N':>12} {'Val %':>10}")
    table_lines.append("-" * 88)

    for b in bucket_order:
        tot_n = total_counts.get(b, 0)
        tot_pct = (tot_n / total_s1 * 100) if total_s1 else 0.0
        tr_n = train_buckets.get(b, 0)
        tr_pct = (tr_n / len(train_ids) * 100) if len(train_ids) else 0.0
        va_n = val_buckets.get(b, 0)
        va_pct = (va_n / len(val_ids) * 100) if len(val_ids) else 0.0
        table_lines.append(f"{b:<16} {tot_n:>12,d} {tot_pct:>9.2f}% {tr_n:>12,d} {tr_pct:>9.2f}% {va_n:>12,d} {va_pct:>9.2f}%")

    table_lines.append("-" * 88)
    table_lines.append(f"{'Total':<16} {total_s1:>12,d} {100.0:>9.2f}% {len(train_ids):>12,d} {100.0:>9.2f}% {len(val_ids):>12,d} {100.0:>9.2f}%")
    table_lines.append("=" * 88)

    verification_table = "\n".join(table_lines)
    print("\n" + verification_table)

    # 6. Partition ground truth rows
    print("\nPartitioning ground truth rows...")
    val_mask = gt_df["source1_entity_id"].isin(val_s1_set)
    val_gt_rows = gt_df[val_mask][["source1_entity_id", "matched_entity_ids"]]
    train_gt_rows = gt_df[~val_mask][["source1_entity_id", "matched_entity_ids"]]

    print(f"  Train Ground Truth Rows: {len(train_gt_rows):,d}")
    print(f"  Val Ground Truth Rows  : {len(val_gt_rows):,d}")

    # 7. Write outputs to disk
    print(f"\nWriting split files to {args.out_dir} ...")

    # train_s1_ids.txt
    p_train_ids = os.path.join(args.out_dir, "train_s1_ids.txt")
    with open(p_train_ids, "w", encoding="utf-8") as f:
        f.write("\n".join(train_ids) + "\n")
    print(f"  [OK] Saved: {p_train_ids} ({len(train_ids):,d} IDs)")

    # val_s1_ids.txt
    p_val_ids = os.path.join(args.out_dir, "val_s1_ids.txt")
    with open(p_val_ids, "w", encoding="utf-8") as f:
        f.write("\n".join(val_ids) + "\n")
    print(f"  [OK] Saved: {p_val_ids} ({len(val_ids):,d} IDs)")

    # train_gt_rows.tsv
    p_train_gt = os.path.join(args.out_dir, "train_gt_rows.tsv")
    train_gt_rows.to_csv(p_train_gt, sep="\t", index=False)
    print(f"  [OK] Saved: {p_train_gt} ({len(train_gt_rows):,d} rows)")

    # val_gt_rows.tsv
    p_val_gt = os.path.join(args.out_dir, "val_gt_rows.tsv")
    val_gt_rows.to_csv(p_val_gt, sep="\t", index=False)
    print(f"  [OK] Saved: {p_val_gt} ({len(val_gt_rows):,d} rows)")

    # split_stats.txt
    p_stats = os.path.join(args.out_dir, "split_stats.txt")
    with open(p_stats, "w", encoding="utf-8") as f:
        f.write("PHASE 3 ENTITY-LEVEL STRATIFIED TRAIN/VAL SPLIT VERIFICATION TABLE\n")
        f.write(f"Source 1 Path      : {args.s1}\n")
        f.write(f"Ground Truth Path  : {args.gt}\n")
        f.write(f"Validation Fraction: {args.val_frac} ({args.val_frac * 100:.1f}%)\n")
        f.write(f"Random Seed        : {args.seed}\n\n")
        f.write(verification_table + "\n\n")
        f.write(f"Train S1 Entities  : {len(train_ids):,d}\n")
        f.write(f"Val S1 Entities    : {len(val_ids):,d}\n")
        f.write(f"Overlap Count      : {len(overlap):,d}\n")
        f.write(f"Total S1 Entities  : {total_s1:,d}\n")
    print(f"  [OK] Saved: {p_stats}")

    elapsed = time.time() - t0
    print(f"\nAll operations completed successfully in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
