"""
Phase 4: Candidate Pairs Generation for Train and Validation Splits.

Loads:
  - splits/train_s1_ids.txt & splits/val_s1_ids.txt
  - dataset/train/train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv

Builds:
  - CountryPartitionBlocking (reusing blocking.py & normalization.py) across India and US

Generates & Writes:
  - candidate_pairs_train.tsv (columns: source1_entity_id, candidate_entity_id)
  - candidate_pairs_val.tsv   (columns: source1_entity_id, candidate_entity_id)

Reports Diagnostics:
  - Overall blocking recall on ground truth
  - Total candidate pairs generated
  - Mean and median candidate set size per S1 entity
  - Reduction ratio vs full Cartesian comparison space
"""

import os
import sys
import gc
import time
import argparse
from typing import Set, Dict, List, Tuple
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "code", "business_entity_resolution", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from blocking import CountryPartitionBlocking


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Candidate Pairs for Train/Val Splits")
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(BASE_DIR, "dataset", "train"),
                        help="Path to train dataset directory")
    parser.add_argument("--splits_dir", type=str, default=os.path.join(BASE_DIR, "splits"),
                        help="Path to splits directory containing train_s1_ids.txt and val_s1_ids.txt")
    parser.add_argument("--out_dir", type=str, default=BASE_DIR,
                        help="Output directory for candidate_pairs_train.tsv and candidate_pairs_val.tsv")
    parser.add_argument("--countries", type=str, default="India,US",
                        help="Comma-separated country list to process (e.g. 'India,US', 'India', or 'US')")
    parser.add_argument("--name_cap", type=int, default=2000, help="Name token posting list cap")
    parser.add_argument("--digit_cap", type=int, default=5000, help="Digit token posting list cap")
    parser.add_argument("--word_cap", type=int, default=2000, help="Address Latin word posting list cap")
    parser.add_argument("--chunksize", type=int, default=500000, help="Chunk size for streaming TSVs")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing output files instead of overwriting (useful when running countries separately)")
    parser.add_argument("--sample_s1", type=int, default=None,
                        help="Optional S1 sample limit for quick diagnostic tests")
    return parser.parse_args()


def load_id_set(path: str) -> Set[str]:
    """Loads a set of entity IDs from a plain text file (one ID per line)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_ground_truth_map(gt_path: str) -> Dict[str, List[str]]:
    """Loads ground truth mapping: source1_entity_id -> list of matched_entity_ids."""
    if not os.path.isfile(gt_path):
        print(f"Warning: Ground truth file {gt_path} not found. Recall stats will be skipped.")
        return {}
    gt_map = {}
    for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""], chunksize=500000):
        for row in chunk.itertuples(index=False):
            m_str = str(row.matched_entity_ids) if pd.notna(row.matched_entity_ids) else ""
            if m_str and m_str.strip() and m_str.lower() not in ("nan", "null"):
                gt_map[str(row.source1_entity_id)] = [x.strip() for x in m_str.split(",") if x.strip()]
            else:
                gt_map[str(row.source1_entity_id)] = []
    return gt_map


def main():
    t_start = time.time()
    args = parse_args()

    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("PHASE 4: CANDIDATE PAIRS GENERATION (TRAIN & VAL SPLITS)")
    print("=" * 80)
    print(f"Dataset Dir        : {args.dataset_dir}")
    print(f"Splits Dir         : {args.splits_dir}")
    print(f"Output Dir         : {args.out_dir}")
    print(f"Target Countries   : {countries}")
    print(f"Index Caps         : Name={args.name_cap}, Digit={args.digit_cap}, Word={args.word_cap}")
    print(f"Append Mode        : {args.append}\n")

    # 1. Load Train and Val S1 ID sets
    p_train_ids = os.path.join(args.splits_dir, "train_s1_ids.txt")
    p_val_ids = os.path.join(args.splits_dir, "val_s1_ids.txt")
    print("Loading entity ID splits...")
    train_s1_set = load_id_set(p_train_ids)
    val_s1_set = load_id_set(p_val_ids)
    print(f"  Train S1 IDs: {len(train_s1_set):,d}")
    print(f"  Val S1 IDs  : {len(val_s1_set):,d}")

    # 2. Load Ground Truth for recall measurement
    gt_path = os.path.join(args.dataset_dir, "train_ground_truth.tsv")
    print(f"\nLoading ground truth from {gt_path}...")
    gt_map = load_ground_truth_map(gt_path)
    print(f"  Loaded GT mapping for {len(gt_map):,} S1 entities.")

    # Paths for output files
    out_train_path = os.path.join(args.out_dir, "candidate_pairs_train.tsv")
    out_val_path = os.path.join(args.out_dir, "candidate_pairs_val.tsv")

    # Open output file handles
    file_mode = "a" if args.append else "w"
    f_train = open(out_train_path, file_mode, encoding="utf-8", buffering=1024 * 1024)
    f_val = open(out_val_path, file_mode, encoding="utf-8", buffering=1024 * 1024)

    # Write headers if starting fresh
    if not args.append:
        f_train.write("source1_entity_id\tcandidate_entity_id\n")
        f_val.write("source1_entity_id\tcandidate_entity_id\n")

    # Global tracking across countries
    stats = {
        "train": {
            "s1_count": 0,
            "cand_pairs": 0,
            "cand_counts": [],
            "true_pairs": 0,
            "recalled_pairs": 0,
            "total_possible_comparisons": 0
        },
        "val": {
            "s1_count": 0,
            "cand_pairs": 0,
            "cand_counts": [],
            "true_pairs": 0,
            "recalled_pairs": 0,
            "total_possible_comparisons": 0
        }
    }

    BUFFER_SIZE = 100000

    # 3. Process each country partition
    for country in countries:
        t_c_start = time.time()
        print("\n" + "-" * 80)
        print(f"PROCESSING COUNTRY PARTITION: {country}")
        print("-" * 80)

        blocking_engine = CountryPartitionBlocking(
            country=country,
            name_cap=args.name_cap,
            digit_cap=args.digit_cap,
            word_cap=args.word_cap
        )

        # Index train_source2 for this country
        s2_path = os.path.join(args.dataset_dir, "train_source2.tsv")
        print(f"Streaming and indexing {country} records from train_source2.tsv...")
        s2_count = 0
        for chunk in pd.read_csv(s2_path, sep="\t", dtype=str,
                                 usecols=["entity_id", "business_name", "business_address", "country"],
                                 keep_default_na=False, na_values=[""], chunksize=args.chunksize):
            c_chunk = chunk[chunk["country"] == country]
            if not c_chunk.empty:
                s2_count += len(c_chunk)
                blocking_engine.index_records(c_chunk)

        # Index train_source3 for this country
        s3_path = os.path.join(args.dataset_dir, "train_source3.tsv")
        print(f"Streaming and indexing {country} records from train_source3.tsv...")
        s3_count = 0
        for chunk in pd.read_csv(s3_path, sep="\t", dtype=str,
                                 usecols=["entity_id", "business_name", "business_address", "country"],
                                 keep_default_na=False, na_values=[""], chunksize=args.chunksize):
            c_chunk = chunk[chunk["country"] == country]
            if not c_chunk.empty:
                s3_count += len(c_chunk)
                blocking_engine.index_records(c_chunk)

        print(f"Indexed {s2_count:,d} S2 records and {s3_count:,d} S3 records for {country}.")
        target_pool_size = s2_count + s3_count

        # Finalize indices
        print(f"Finalizing inverted indices for {country}...")
        blocking_engine.finalize()

        # Query candidates for S1 records in this country
        s1_path = os.path.join(args.dataset_dir, "train_source1.tsv")
        print(f"\nStreaming S1 records for {country} and generating candidate pairs...")

        train_buf = []
        val_buf = []
        c_s1_processed = 0

        for chunk in pd.read_csv(s1_path, sep="\t", dtype=str,
                                 usecols=["entity_id", "business_name", "business_address", "country"],
                                 keep_default_na=False, na_values=[""], chunksize=args.chunksize):
            c_chunk = chunk[chunk["country"] == country]
            if c_chunk.empty:
                continue

            for row in c_chunk.itertuples(index=False):
                s1_id = str(row.entity_id)

                # Determine split
                is_train = s1_id in train_s1_set
                is_val = s1_id in val_s1_set

                if not is_train and not is_val:
                    continue

                split_key = "train" if is_train else "val"
                buf = train_buf if is_train else val_buf
                fh = f_train if is_train else f_val

                s1_name = str(row.business_name) if pd.notna(row.business_name) else ""
                s1_addr = str(row.business_address) if pd.notna(row.business_address) else ""

                # Query candidates (direct union of passes)
                candidates = blocking_engine.get_candidates(s1_name, s1_addr)
                n_cands = len(candidates)

                # Buffer pairs
                for cid in sorted(candidates):
                    buf.append(f"{s1_id}\t{cid}\n")
                    if len(buf) >= BUFFER_SIZE:
                        fh.writelines(buf)
                        buf.clear()

                # Track metrics
                st = stats[split_key]
                st["s1_count"] += 1
                st["cand_pairs"] += n_cands
                st["cand_counts"].append(n_cands)
                st["total_possible_comparisons"] += target_pool_size

                # Ground truth recall check
                true_matches = gt_map.get(s1_id, [])
                if true_matches:
                    st["true_pairs"] += len(true_matches)
                    for mid in true_matches:
                        if mid in candidates:
                            st["recalled_pairs"] += 1

                c_s1_processed += 1
                if args.sample_s1 and c_s1_processed >= args.sample_s1:
                    break

            if args.sample_s1 and c_s1_processed >= args.sample_s1:
                break

        # Flush remaining buffers
        if train_buf:
            f_train.writelines(train_buf)
            train_buf.clear()
        if val_buf:
            f_val.writelines(val_buf)
            val_buf.clear()

        f_train.flush()
        f_val.flush()

        t_c_elapsed = time.time() - t_c_start
        print(f"Finished {country} partition in {t_c_elapsed:.1f}s ({t_c_elapsed/60:.2f} min). Processed {c_s1_processed:,d} S1 records.")

        # Clean up memory
        del blocking_engine
        gc.collect()

    f_train.close()
    f_val.close()

    total_wall_clock = time.time() - t_start

    # 4. Final Diagnostics Report
    print("\n" + "=" * 80)
    print("PHASE 4: BLOCKING & CANDIDATE GENERATION SUMMARY REPORT")
    print("=" * 80)

    hdr = f"{'Split':<10} {'S1 Count':>12} {'Candidate Pairs':>18} {'Mean Cands':>12} {'Median':>8} {'Recall %':>12} {'Recalled/Total':>20} {'Reduction %':>14}"
    print(hdr)
    print("-" * len(hdr))

    for split_name in ["train", "val"]:
        st = stats[split_name]
        s1_n = st["s1_count"]
        cand_n = st["cand_pairs"]
        c_arr = np.array(st["cand_counts"], dtype=np.int32) if st["cand_counts"] else np.array([0])
        mean_c = float(c_arr.mean()) if s1_n else 0.0
        med_c = float(np.median(c_arr)) if s1_n else 0.0

        tp = st["true_pairs"]
        rec = st["recalled_pairs"]
        rec_pct = (rec / tp * 100) if tp else 0.0
        rec_str = f"{rec:,d}/{tp:,d}"

        tot_possible = st["total_possible_comparisons"]
        red_pct = ((1.0 - (cand_n / tot_possible)) * 100) if tot_possible else 100.0

        print(f"{split_name:<10} {s1_n:>12,d} {cand_n:>18,d} {mean_c:>12.2f} {med_c:>8.1f} {rec_pct:>11.3f}% {rec_str:>20} {red_pct:>13.6f}%")

    print("-" * len(hdr))
    print(f"\nFiles Written:")
    print(f"  - Train: {out_train_path} ({stats['train']['cand_pairs']:,d} pairs)")
    print(f"  - Val  : {out_val_path} ({stats['val']['cand_pairs']:,d} pairs)")
    print(f"\nTotal Wall-Clock Time: {total_wall_clock:.1f}s ({total_wall_clock/60:.2f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
