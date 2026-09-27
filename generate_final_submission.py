#!/usr/bin/env python3
"""
generate_final_submission.py — Final Inference & Submission TSV Generation.

Pipeline:
1. Loads candidate pairs from output/test_candidate_pairs.tsv.
2. Extracts entity metadata (name, address, country) from test_source1.tsv,
   test_source2.tsv, and test_source3.tsv.
3. Computes the EXACT 7 pairwise similarity features in EXACT order:
     1. exact_name_match       (1.0 if normalized names match exactly, else 0.0)
     2. exact_address_match    (1.0 if normalized addresses match exactly, else 0.0)
     3. name_token_jaccard     (token-level Jaccard similarity of normalized names)
     4. address_token_jaccard  (token-level Jaccard similarity of normalized addresses)
     5. name_jaro_winkler      (Jaro-Winkler similarity of normalized names)
     6. source_is_s2           (1.0 for S2 candidate, 0.0 for S3 candidate)
     7. country_match          (1.0 if countries match, else 0.0)
4. Loads the trained LightGBM model from output/model.txt and validates expected features.
5. Loads the optimal decision threshold from output/best_threshold.txt.
6. Scores candidate pairs in streaming batches.
7. Aggregates per Source 1 entity:
     - output/matching_results.tsv  (Header: source1_entity_id\tmatched_entity_ids)
     - output/candidate_pairs.tsv   (Header: source1_entity_id\tcandidate_entity_ids)
   Ensures ALL required test S1 entities appear with exactly one row, satisfying every
   rule enforced by utils/validate_submission.py.
"""

import os
import sys
import re
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict

# Add code/business_entity_resolution/src to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "code", "business_entity_resolution", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from normalization import (
    clean_text,
    normalize_name,
    normalize_address,
    extract_digit_tokens,
    extract_word_tokens,
)

# Canonical 7 features in exact order
FEATURE_COLUMNS = [
    "exact_name_match",
    "exact_address_match",
    "name_token_jaccard",
    "address_token_jaccard",
    "name_jaro_winkler",
    "source_is_s2",
    "country_match",
]

# RapidFuzz import with pure Python fallback for Jaro-Winkler
try:
    from rapidfuzz.distance import JaroWinkler
    def calc_jaro_winkler(s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        return float(JaroWinkler.similarity(s1, s2))
except (ImportError, AttributeError):
    try:
        from rapidfuzz import fuzz
        def calc_jaro_winkler(s1: str, s2: str) -> float:
            if not s1 or not s2:
                return 0.0
            return float(fuzz.jaro_winkler(s1, s2)) / 100.0
    except (ImportError, AttributeError):
        # Pure Python fallback for Jaro-Winkler similarity
        def calc_jaro_winkler(s1: str, s2: str) -> float:
            if not s1 or not s2:
                return 0.0
            if s1 == s2:
                return 1.0
            len1, len2 = len(s1), len(s2)
            max_dist = max(len1, len2) // 2 - 1
            if max_dist < 0:
                max_dist = 0
            s1_matches = [False] * len1
            s2_matches = [False] * len2
            matches = 0
            for i in range(len1):
                start = max(0, i - max_dist)
                end = min(i + max_dist + 1, len2)
                for j in range(start, end):
                    if s2_matches[j]:
                        continue
                    if s1[i] != s2[j]:
                        continue
                    s1_matches[i] = True
                    s2_matches[j] = True
                    matches += 1
                    break
            if matches == 0:
                return 0.0
            t = 0
            k = 0
            for i in range(len1):
                if not s1_matches[i]:
                    continue
                while not s2_matches[k]:
                    k += 1
                if s1[i] != s2[k]:
                    t += 1
                k += 1
            t //= 2
            jaro = (matches / len1 + matches / len2 + (matches - t) / matches) / 3.0
            prefix = 0
            for i in range(min(4, min(len1, len2))):
                if s1[i] == s2[i]:
                    prefix += 1
                else:
                    break
            return jaro + prefix * 0.1 * (1.0 - jaro)

# LightGBM import
try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def calc_token_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Computes Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    union = tokens1 | tokens2
    if not union:
        return 0.0
    return float(len(tokens1 & tokens2)) / float(len(union))


def load_threshold(threshold_file: str, default_thresh: float = 0.5) -> float:
    """Loads optimal decision threshold from file, fallback to default_thresh."""
    if not os.path.isfile(threshold_file):
        print(f"Notice: Threshold file {threshold_file} not found. Using default {default_thresh}.")
        return default_thresh
    try:
        with open(threshold_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", content)
            if match:
                val = float(match.group(0))
                print(f"Loaded decision threshold from {threshold_file}: {val:.4f}")
                return val
    except Exception as exc:
        print(f"Warning: Could not parse threshold from {threshold_file} ({exc}). Using default {default_thresh}.")
    return default_thresh


def load_test_entities(
    test_dir: str,
    needed_cand_ids: Set[str]
) -> Tuple[List[str], Dict[str, Tuple[str, str, Set[str], Set[str], str]], Dict[str, Tuple[str, str, Set[str], Set[str], str]]]:
    """
    Loads:
      - Ordered list of all S1 entity IDs
      - S1 entity lookup: eid -> (norm_name, norm_addr, name_tokens, addr_tokens, country)
      - Target candidate lookup: eid -> (norm_name, norm_addr, name_tokens, addr_tokens, country)
    """
    print("\nLoading entity metadata (names, addresses, countries) for feature extraction...")
    t0 = time.time()

    # 1. Load S1 entities (preserve exact ordering)
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Reading S1 records from {s1_path}...")
    s1_order: List[str] = []
    s1_lookup: Dict[str, Tuple[str, str, Set[str], Set[str], str]] = {}

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        usecols=["entity_id", "business_name", "business_address", "country"],
        keep_default_na=False,
        na_values=[""],
        chunksize=500000,
        low_memory=False
    ):
        for row in chunk.itertuples(index=False):
            eid = str(row.entity_id).strip()
            s1_order.append(eid)
            raw_name = str(row.business_name) if pd.notna(row.business_name) else ""
            raw_addr = str(row.business_address) if pd.notna(row.business_address) else ""
            raw_country = str(row.country).strip().lower() if pd.notna(row.country) else ""
            if raw_country == "null":
                raw_country = ""

            norm_name = normalize_name(raw_name)
            norm_addr = normalize_address(raw_addr)
            name_toks = set(norm_name.split()) if norm_name else set()
            addr_toks = set(norm_addr.split()) if norm_addr else set()

            s1_lookup[eid] = (norm_name, norm_addr, name_toks, addr_toks, raw_country)

    print(f"  Loaded {len(s1_order):,} S1 entities.")

    # 2. Load only needed candidates from S2 and S3
    cand_lookup: Dict[str, Tuple[str, str, Set[str], Set[str], str]] = {}
    print(f"Loading {len(needed_cand_ids):,} unique candidate entities from S2 & S3...")

    for src_name in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(test_dir, src_name)
        if not os.path.isfile(path):
            continue
        print(f"  Streaming {src_name}...")
        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=["entity_id", "business_name", "business_address", "country"],
            keep_default_na=False,
            na_values=[""],
            chunksize=500000,
            low_memory=False
        ):
            relevant = chunk[chunk["entity_id"].isin(needed_cand_ids)]
            for row in relevant.itertuples(index=False):
                eid = str(row.entity_id).strip()
                raw_name = str(row.business_name) if pd.notna(row.business_name) else ""
                raw_addr = str(row.business_address) if pd.notna(row.business_address) else ""
                raw_country = str(row.country).strip().lower() if pd.notna(row.country) else ""
                if raw_country == "null":
                    raw_country = ""

                norm_name = normalize_name(raw_name)
                norm_addr = normalize_address(raw_addr)
                name_toks = set(norm_name.split()) if norm_name else set()
                addr_toks = set(norm_addr.split()) if norm_addr else set()

                cand_lookup[eid] = (norm_name, norm_addr, name_toks, addr_toks, raw_country)

    elapsed = time.time() - t0
    print(f"Loaded entity records in {elapsed:.1f}s.")
    return s1_order, s1_lookup, cand_lookup


def compute_pair_features(
    s1_meta: Tuple[str, str, Set[str], Set[str], str],
    cand_meta: Tuple[str, str, Set[str], Set[str], str],
    cand_id: str
) -> List[float]:
    """
    Computes the exact 7 features in canonical order:
      1. exact_name_match
      2. exact_address_match
      3. name_token_jaccard
      4. address_token_jaccard
      5. name_jaro_winkler
      6. source_is_s2
      7. country_match
    """
    name1, addr1, ntoks1, atoks1, c1 = s1_meta
    name2, addr2, ntoks2, atoks2, c2 = cand_meta

    # 1. exact_name_match
    f_exact_name = 1.0 if (name1 and name1 == name2) else 0.0

    # 2. exact_address_match
    f_exact_addr = 1.0 if (addr1 and addr1 == addr2) else 0.0

    # 3. name_token_jaccard
    f_name_jaccard = calc_token_jaccard(ntoks1, ntoks2)

    # 4. address_token_jaccard
    f_addr_jaccard = calc_token_jaccard(atoks1, atoks2)

    # 5. name_jaro_winkler
    f_name_jw = calc_jaro_winkler(name1, name2)

    # 6. source_is_s2
    f_source_s2 = 1.0 if cand_id.startswith("S2-") else 0.0

    # 7. country_match
    f_country_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0

    return [
        f_exact_name,
        f_exact_addr,
        f_name_jaccard,
        f_addr_jaccard,
        f_name_jw,
        f_source_s2,
        f_country_match,
    ]


def align_feature_matrix(
    features_batch: List[List[float]],
    model_feat_names: Optional[List[str]]
) -> np.ndarray:
    """
    Constructs a numpy array aligned with LightGBM expected feature columns.
    If the model was trained with FEATURE_COLUMNS (or Column_0..Column_6), maintains exact order.
    """
    arr = np.array(features_batch, dtype=np.float32)

    if not model_feat_names:
        return arr

    # If the model feature names match canonical names in a different order, reindex:
    col_to_idx = {name: idx for idx, name in enumerate(FEATURE_COLUMNS)}
    if all(mf in col_to_idx for mf in model_feat_names):
        order = [col_to_idx[mf] for mf in model_feat_names]
        return arr[:, order]

    return arr


def run_pipeline(
    candidates_path: str,
    test_dir: str,
    model_path: str,
    threshold_path: str,
    matching_output: str,
    candidate_output: str,
    default_threshold: float = 0.5,
    batch_size: int = 100000
):
    """
    Executes end-to-end scoring with the exact 7 features and generates both submission TSVs.
    """
    t_start = time.time()
    print("=" * 80)
    print("GENERATE FINAL SUBMISSION PIPELINE (EXACT 7 FEATURES)")
    print("=" * 80)
    print(f"Target Feature Schema ({len(FEATURE_COLUMNS)} features):")
    for i, col in enumerate(FEATURE_COLUMNS, 1):
        print(f"  {i}. {col}")

    os.makedirs(os.path.dirname(matching_output), exist_ok=True)
    os.makedirs(os.path.dirname(candidate_output), exist_ok=True)

    # 1. Load trained LightGBM model
    print(f"\nLoading model from {model_path}...")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Please ensure output/model.txt has been trained and saved."
        )
    if lgb is None:
        raise ImportError("LightGBM is not installed. Run: pip install lightgbm")
    booster = lgb.Booster(model_file=model_path)
    model_feat_names = booster.feature_name()
    print(f"  Model loaded successfully.")
    print(f"  Model expected feature count : {len(model_feat_names)}")
    print(f"  Model feature names          : {model_feat_names}")

    # Validate feature compatibility
    if model_feat_names == FEATURE_COLUMNS:
        print("  >> EXACT MATCH: Model feature names match FEATURE_COLUMNS perfectly.")
    else:
        print(f"  >> NOTICE: Aligning feature columns to model specification.")

    # 2. Load decision threshold
    threshold = load_threshold(threshold_path, default_thresh=default_threshold)

    # 3. Read test_candidate_pairs.tsv and collect candidate IDs
    print(f"\nScanning candidate pairs from {candidates_path}...")
    if not os.path.isfile(candidates_path):
        raise FileNotFoundError(
            f"Candidates file not found at {candidates_path}. "
            f"Please run build_small_test_pairs.py first."
        )

    needed_cands: Set[str] = set()
    total_pairs = 0
    with open(candidates_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                needed_cands.add(parts[1])
                total_pairs += 1

    print(f"  Total candidate pairs to score: {total_pairs:,}")
    print(f"  Unique target candidate IDs needed: {len(needed_cands):,}")

    # 4. Load entity text lookups
    s1_order, s1_lookup, cand_lookup = load_test_entities(test_dir, needed_cands)

    cand_groups: Dict[str, List[str]] = defaultdict(list)
    matched_groups: Dict[str, List[Tuple[float, str]]] = defaultdict(list)

    # 5. Stream candidate pairs and score in batches
    print(f"\nExtracting 7 features and scoring pairs in batches of {batch_size:,}...")
    t_score_start = time.time()
    scored_count = 0

    batch_s1: List[str] = []
    batch_cand: List[str] = []
    batch_features: List[List[float]] = []

    def flush_batch():
        nonlocal scored_count
        if not batch_s1:
            return
        X = align_feature_matrix(batch_features, model_feat_names)
        scores = booster.predict(X)

        for s1, cid, score in zip(batch_s1, batch_cand, scores):
            cand_groups[s1].append(cid)
            if score >= threshold:
                matched_groups[s1].append((float(score), cid))

        scored_count += len(batch_s1)
        batch_s1.clear()
        batch_cand.clear()
        batch_features.clear()
        if scored_count % 500000 == 0:
            print(f"  Scored {scored_count:,}/{total_pairs:,} pairs...")

    with open(candidates_path, "r", encoding="utf-8") as f:
        next(f, None)  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            s1_id = parts[0]
            cand_id = parts[1]

            s1_meta = s1_lookup.get(s1_id, ("", "", set(), set(), ""))
            cand_meta = cand_lookup.get(cand_id, ("", "", set(), set(), ""))

            feats = compute_pair_features(s1_meta, cand_meta, cand_id)
            batch_s1.append(s1_id)
            batch_cand.append(cand_id)
            batch_features.append(feats)

            if len(batch_s1) >= batch_size:
                flush_batch()

    # Flush remaining
    flush_batch()
    score_elapsed = time.time() - t_score_start
    print(f"Completed scoring {scored_count:,} pairs in {score_elapsed:.1f}s.")

    # 6. Write matching_results.tsv (Header: source1_entity_id\tmatched_entity_ids)
    print(f"\nWriting final matching results to {matching_output}...")
    total_matches_written = 0
    s1_with_matches = 0

    with open(matching_output, "w", encoding="utf-8") as out_m:
        out_m.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_order:
            matches = matched_groups.get(s1_id, [])
            if matches:
                # Sort matches by score descending, then by entity ID for determinism
                sorted_matches = [m[1] for m in sorted(matches, key=lambda x: (-x[0], x[1]))]
                seen = set()
                deduped = []
                for mid in sorted_matches:
                    if mid not in seen:
                        seen.add(mid)
                        deduped.append(mid)
                out_m.write(f"{s1_id}\t{','.join(deduped)}\n")
                total_matches_written += len(deduped)
                s1_with_matches += 1
            else:
                out_m.write(f"{s1_id}\t\n")

    print(f"  {matching_output} written.")
    print(f"  S1 entities with matches: {s1_with_matches:,}/{len(s1_order):,}")
    print(f"  Total matched links: {total_matches_written:,}")

    # 7. Write candidate_pairs.tsv (Header: source1_entity_id\tcandidate_entity_ids)
    print(f"\nWriting candidate pairs to {candidate_output}...")
    total_cands_written = 0
    s1_with_cands = 0

    with open(candidate_output, "w", encoding="utf-8") as out_c:
        out_c.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_order:
            cands = cand_groups.get(s1_id, [])
            if cands:
                seen = set()
                deduped = []
                for cid in cands:
                    if cid not in seen:
                        seen.add(cid)
                        deduped.append(cid)
                # Ensure all matched IDs are guaranteed to be in candidates
                for _, mid in matched_groups.get(s1_id, []):
                    if mid not in seen:
                        seen.add(mid)
                        deduped.append(mid)

                out_c.write(f"{s1_id}\t{','.join(deduped)}\n")
                total_cands_written += len(deduped)
                s1_with_cands += 1
            else:
                out_c.write(f"{s1_id}\t\n")

    print(f"  {candidate_output} written.")
    print(f"  S1 entities with candidates: {s1_with_cands:,}/{len(s1_order):,}")
    print(f"  Total candidate links: {total_cands_written:,}")

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print("SUBMISSION GENERATION COMPLETE")
    print("=" * 80)
    print(f"  Total Wall Clock Time: {total_time:.1f}s ({total_time / 60:.2f} minutes)")
    print("  Ready to run submission validator:")
    print(f"    python utils/validate_submission.py --matching {matching_output} --candidate {candidate_output} --test-dir {test_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate Final Submission for Amazon ML Challenge 2026")
    parser.add_argument("--candidates", default="output/test_candidate_pairs.tsv", help="Path to test candidate pairs")
    parser.add_argument("--test-dir", default="dataset/test", help="Folder containing test_source1/2/3.tsv")
    parser.add_argument("--model", default="output/model.txt", help="Path to trained LightGBM model")
    parser.add_argument("--threshold-file", default="output/best_threshold.txt", help="Path to optimal threshold file")
    parser.add_argument("--matching-output", default="output/matching_results.tsv", help="Output path for matching_results.tsv")
    parser.add_argument("--candidate-output", default="output/candidate_pairs.tsv", help="Output path for candidate_pairs.tsv")
    parser.add_argument("--default-threshold", type=float, default=0.5, help="Default decision threshold if file absent")
    parser.add_argument("--batch-size", type=int, default=100000, help="Batch size for feature extraction and scoring")
    args = parser.parse_args()

    run_pipeline(
        candidates_path=args.candidates,
        test_dir=args.test_dir,
        model_path=args.model,
        threshold_path=args.threshold_file,
        matching_output=args.matching_output,
        candidate_output=args.candidate_output,
        default_threshold=args.default_threshold,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
