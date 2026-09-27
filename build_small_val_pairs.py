#!/usr/bin/env python3
"""
build_small_val_pairs.py — Fast Lexically-Aware Candidate Pair Generator (Validation 500 Subset).

Generates validation pairs for the 500 S1 entity subset:
  - All ground-truth matching S2/S3 entities as positive pairs (label=1)
  - Hard negative pairs (label=0) generated via fast same-country inverted index blocking
    with lexical overlap (name tokens & address digits), not fully random
  - Max 10 negatives per positive pair (or up to 10 for singletons)
  - Fully deterministic with seed=42
  - Column schema: source1_entity_id\tcandidate_entity_id\tsource\tlabel
  - Writes to output/small_val_pairs.tsv
  - Prints comprehensive summary statistics:
      * total S1 entities
      * positive pairs, negative pairs, total pairs, positive %
      * India / US distribution
      * Source 2 / Source 3 distribution
"""

import os
import sys
import time
import random
import re
import argparse
from typing import Dict, Set, List, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "code", "business_entity_resolution", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from normalization import normalize_name, normalize_address, extract_digit_tokens

STOPWORDS = {"and", "the", "of", "in", "for", "at", "to", "a", "an", "on", "by", "co", "ltd", "inc", "llc"}


def extract_lexical_tokens(name: str, address: str) -> Set[str]:
    """Extracts fast lexical tokens for blocking (name words >= 3 chars, address digits >= 3)."""
    tokens = set()
    norm_n = normalize_name(name)
    if norm_n:
        for t in norm_n.split():
            if len(t) >= 3 and t not in STOPWORDS:
                tokens.add(t)

    # Address digits (PINs, street numbers)
    digits = extract_digit_tokens(address)
    for d in digits:
        tokens.add(f"#d_{d}")

    return tokens


def run_build_small_val_pairs(
    subset_ids_path: str = os.path.join(BASE_DIR, "splits", "val_subset_500_ids.txt"),
    dataset_dir: str = os.path.join(BASE_DIR, "dataset", "train"),
    output_path: str = os.path.join(BASE_DIR, "output", "small_val_pairs.tsv"),
    max_neg_ratio: int = 10,
    seed: int = 42,
    chunksize: int = 500000
):
    t_start = time.time()
    print("=" * 80)
    print("STEP C: BUILD SMALL VALIDATION PAIRS (500 S1 SUBSET)")
    print("=" * 80)
    print(f"Subset File : {subset_ids_path}")
    print(f"Dataset Dir : {dataset_dir}")
    print(f"Output Path : {output_path}")
    print(f"Max Neg / Pos: {max_neg_ratio} (seed={seed})\n")

    # 1. Load Subset S1 IDs
    if not os.path.isfile(subset_ids_path):
        raise FileNotFoundError(
            f"Subset IDs file not found: {subset_ids_path}. Run create_subsets.py first."
        )

    with open(subset_ids_path, "r", encoding="utf-8") as f:
        sampled_s1_ids = [line.strip() for line in f if line.strip()]

    target_s1_set = set(sampled_s1_ids)
    print(f"Loaded {len(sampled_s1_ids):,} target validation S1 entity IDs.")

    # 2. Load Metadata for Target S1 Entities
    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    print(f"Reading target S1 metadata from {s1_path}...")
    s1_metadata: Dict[str, Tuple[str, str, str, Set[str]]] = {}
    needed_tokens_by_country: Dict[str, Set[str]] = defaultdict(set)

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        usecols=["entity_id", "business_name", "business_address", "country"],
        keep_default_na=False,
        na_values=[""],
        chunksize=chunksize,
        low_memory=False
    ):
        rel = chunk[chunk["entity_id"].isin(target_s1_set)]
        for row in rel.itertuples(index=False):
            eid = str(row.entity_id)
            name = str(row.business_name) if pd.notna(row.business_name) else ""
            addr = str(row.business_address) if pd.notna(row.business_address) else ""
            country = str(row.country).strip() if pd.notna(row.country) else "Unknown"

            tokens = extract_lexical_tokens(name, addr)
            s1_metadata[eid] = (name, addr, country, tokens)
            needed_tokens_by_country[country].update(tokens)

    print(f"  Loaded metadata for {len(s1_metadata):,} S1 entities.")
    for c, toks in needed_tokens_by_country.items():
        print(f"    - {c}: {len(toks):,} active lexical tokens")

    # 3. Load Ground Truth Positives
    gt_path = os.path.join(dataset_dir, "train_ground_truth.tsv")
    print(f"\nLoading ground truth matches from {gt_path}...")
    gt_matches: Dict[str, List[str]] = defaultdict(list)
    gt_positive_pairs = []

    for chunk in pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        usecols=["source1_entity_id", "matched_entity_ids"],
        keep_default_na=False,
        na_values=[""],
        chunksize=chunksize,
        low_memory=False
    ):
        rel = chunk[chunk["source1_entity_id"].isin(target_s1_set)]
        for row in rel.itertuples(index=False):
            s1_id = str(row.source1_entity_id)
            raw_m = str(row.matched_entity_ids) if pd.notna(row.matched_entity_ids) else ""
            matches = [x.strip() for x in raw_m.split(",") if x.strip()]
            for mid in matches:
                src_tag = "S2" if mid.startswith("S2-") else "S3"
                gt_matches[s1_id].append(mid)
                gt_positive_pairs.append((s1_id, mid, src_tag, 1))

    print(f"  Found {len(gt_positive_pairs):,} positive match pairs across {len(s1_metadata):,} S1 entities.")

    # 4. Stream S2 and S3 to Build Fast Candidate Negative Index
    print("\nIndexing S2 and S3 for same-country lexical negative generation...")
    POSTING_CAP = 500
    inverted_index: Dict[str, Dict[str, List[str]]] = {
        c: defaultdict(list) for c in needed_tokens_by_country
    }
    candidate_country_map: Dict[str, str] = {}

    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(dataset_dir, src_file)
        if not os.path.isfile(path):
            continue
        print(f"  Streaming {src_file}...")
        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=["entity_id", "business_name", "business_address", "country"],
            keep_default_na=False,
            na_values=[""],
            chunksize=chunksize,
            low_memory=False
        ):
            for row in chunk.itertuples(index=False):
                country = str(row.country).strip() if pd.notna(row.country) else "Unknown"
                active_tokens = needed_tokens_by_country.get(country)
                if not active_tokens:
                    continue

                eid = str(row.entity_id)
                name = str(row.business_name) if pd.notna(row.business_name) else ""
                addr = str(row.business_address) if pd.notna(row.business_address) else ""

                row_toks = extract_lexical_tokens(name, addr)
                shared = row_toks & active_tokens
                if shared:
                    candidate_country_map[eid] = country
                    c_idx = inverted_index[country]
                    for tok in shared:
                        plist = c_idx[tok]
                        if len(plist) < POSTING_CAP:
                            plist.append(eid)

    total_postings = sum(sum(len(v) for v in idx.values()) for idx in inverted_index.values())
    print(f"  Inverted negative index built with {total_postings:,} total postings.")

    # 5. Generate Lexically-Aware Negatives
    print("\nGenerating negatives with lexical overlap per S1 entity...")
    rng = random.Random(seed)
    negative_pairs = []

    for s1_id in sampled_s1_ids:
        name, addr, country, s1_toks = s1_metadata.get(s1_id, ("", "", "Unknown", set()))
        true_positives = set(gt_matches.get(s1_id, []))
        num_pos = len(true_positives)

        n_neg_target = max(10, num_pos * max_neg_ratio)

        c_idx = inverted_index.get(country, {})
        cand_overlap_counts = Counter()
        for tok in s1_toks:
            for cid in c_idx.get(tok, []):
                if cid not in true_positives and cid != s1_id:
                    cand_overlap_counts[cid] += 1

        if cand_overlap_counts:
            by_score = defaultdict(list)
            for cid, score in cand_overlap_counts.items():
                by_score[score].append(cid)

            selected_cands = []
            for score in sorted(by_score.keys(), reverse=True):
                group = sorted(by_score[score])
                rng.shuffle(group)
                needed = n_neg_target - len(selected_cands)
                selected_cands.extend(group[:needed])
                if len(selected_cands) >= n_neg_target:
                    break

            for cid in selected_cands:
                src_tag = "S2" if cid.startswith("S2-") else "S3"
                negative_pairs.append((s1_id, cid, src_tag, 0))

    print(f"  Generated {len(negative_pairs):,} negative pairs.")

    # 6. Combine and Shuffle Deterministically
    all_pairs = gt_positive_pairs + negative_pairs
    rng.shuffle(all_pairs)

    # 7. Write output/small_val_pairs.tsv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\nWriting {len(all_pairs):,} pairs to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_id\tsource\tlabel\n")
        for s1, cand, src, lbl in all_pairs:
            f.write(f"{s1}\t{cand}\t{src}\t{lbl}\n")

    # 8. Compute and Print Summary Statistics
    total_pairs = len(all_pairs)
    n_pos = len(gt_positive_pairs)
    n_neg = len(negative_pairs)
    pos_pct = (n_pos / total_pairs * 100) if total_pairs else 0.0

    country_counts = Counter()
    for s1, _, _, _ in all_pairs:
        country_counts[s1_metadata[s1][2]] += 1

    src_counts = Counter(p[2] for p in all_pairs)

    print("\n" + "=" * 80)
    print("SMALL VALIDATION PAIRS SUMMARY REPORT")
    print("=" * 80)
    print(f"  Total S1 Entities Processed: {len(s1_metadata):,}")
    print(f"  Total Positive Pairs (1)   : {n_pos:>8,d} ({pos_pct:>5.2f}%)")
    print(f"  Total Negative Pairs (0)   : {n_neg:>8,d} ({100 - pos_pct:>5.2f}%)")
    print(f"  Total Validation Pairs     : {total_pairs:>8,d}")
    print("-" * 50)
    print("  Country Distribution of Pairs:")
    for c_name, c_cnt in country_counts.most_common():
        pct = c_cnt / total_pairs * 100 if total_pairs else 0
        print(f"    - {c_name:<10}: {c_cnt:>8,d} ({pct:>5.2f}%)")
    print("-" * 50)
    print("  Source Distribution of Candidates:")
    for s_tag, s_cnt in src_counts.most_common():
        pct = s_cnt / total_pairs * 100 if total_pairs else 0
        print(f"    - {s_tag:<10}: {s_cnt:>8,d} ({pct:>5.2f}%)")
    print("=" * 80)

    elapsed = time.time() - t_start
    print(f"Validation pair generation complete in {elapsed:.1f}s ({elapsed / 60:.2f} min).")


def main():
    parser = argparse.ArgumentParser(description="Build small validation candidate pairs (500 subset)")
    parser.add_argument("--subset", default=os.path.join(BASE_DIR, "splits", "val_subset_500_ids.txt"),
                        help="Path to subset entity IDs file")
    parser.add_argument("--dataset-dir", default=os.path.join(BASE_DIR, "dataset", "train"),
                        help="Train dataset directory")
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "output", "small_val_pairs.tsv"),
                        help="Output path for pairs TSV")
    parser.add_argument("--max-neg-ratio", type=int, default=10, help="Max negatives per positive")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_build_small_val_pairs(
        subset_ids_path=args.subset,
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        max_neg_ratio=args.max_neg_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
