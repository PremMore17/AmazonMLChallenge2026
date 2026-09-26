"""
Phase 2 & 3 Runner Script: Normalization + Index-Based Blocking Evaluation on TRAIN.

Executes:
  Step 1: Normalization sanity check (2,000 samples per source, 15 before/after examples).
  Step 2: Inverted index construction with frequency capping (posting stats, capped keys/postings).
  Step 3: Per-S1 candidate generation across country partitions.
  Step 4: Comprehensive recall measurement on TRAIN ground truth (all, same-script, cross-script),
          recall breakdown by pass (name/digit/word), candidate count distribution (mean, p50, p90, p99, max),
          and runtime.

Saves full output to phase2_3_output.txt and prints to console.
"""
from typing import Optional, List, Dict, Set, Tuple

import os
import sys
import time
import argparse
import random
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Optional

# Ensure code/business_entity_resolution is in sys.path
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
    latin_fraction,
    is_latin_majority
)
from blocking import CountryPartitionBlocking, STOPWORDS
from data_loader import load_tsv

OUTPUT_LOG = os.path.join(BASE_DIR, "phase2_3_output.txt")

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


def run_step1_normalization_check(dataset_dir: str):
    """Step 1: Run normalization on 2000-row sample per source, print 15 before/after examples."""
    print("=" * 80)
    print("STEP 1: NORMALIZATION SANITY CHECK (2,000 ROWS PER SOURCE)")
    print("=" * 80)

    # Explicit verification of targeted fixes: NaN handling, Diacritic Folding, Dotted Acronyms, Indic Script Preservation, Standalone Null
    print("\n--- Targeted Unit Verification of Fixes ---")
    unit_tests = [
        ("Mált Inc", normalize_name, "malt"),
        ("Mált L.L.C.", normalize_name, "malt"),
        ("Sweet Barbershop L.L.C.", normalize_name, "sweet barbershop"),
        ("Fresh Truist U.S.A. Inc.", normalize_name, "fresh truist usa"),
        ("Láne St.", normalize_address, "lane street"),
        ("210 Driver Street, NULL, Drham, North Carolina", normalize_address, "210 driver street drham north carolina"),
        ("बाबा ट्रेडिंग प्राइवेट लिमिटेड", normalize_name, "बाबा ट्रेडिंग प्राइवेट लिमिटेड"),
        ("ಡ್ರೀಮ್ ಟೆಕ್ನಾಲಜಿ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", normalize_name, "ಡ್ರೀಮ್ ಟೆಕ್ನಾಲಜಿ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್"),
        (float("nan"), clean_text, ""),
        ("null", clean_text, ""),
        (None, clean_text, "")
    ]
    for inp, fn, expected in unit_tests:
        out = fn(inp)
        status = "PASS" if out == expected else f"FAIL (got {out!r})"
        print(f"  [{status}] {fn.__name__}({inp!r}) -> {out!r}")


    for src in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(dataset_dir, src)
        if not os.path.isfile(path):
            continue
        print(f"\n--- Sampling 2,000 rows from {src} ---")
        df = pd.read_csv(path, sep="\t", dtype=str, nrows=20000, keep_default_na=False, na_values=[""])
        sample = df.sample(min(2000, len(df)), random_state=42)

        # 15 examples for normalize_name
        name_examples = []
        for name in sample["business_name"]:
            if pd.isna(name) or not name or str(name).strip().lower() == "null":
                continue
            norm = normalize_name(name)
            if norm != str(name).strip():
                name_examples.append((str(name), norm))

        print(f"\n[normalize_name] 15 Before / After examples from {src}:")
        for orig, norm in random.sample(name_examples, min(15, len(name_examples))):
            print(f"  BEFORE: {orig!r}")
            print(f"  AFTER : {norm!r}")

        # 15 examples for normalize_address
        addr_examples = []
        for addr in sample["business_address"]:
            if pd.isna(addr) or not addr or str(addr).strip().lower() == "null":
                continue
            norm = normalize_address(addr)
            if norm != str(addr).strip():
                addr_examples.append((str(addr), norm))

        print(f"\n[normalize_address] 15 Before / After examples from {src}:")
        for orig, norm in random.sample(addr_examples, min(15, len(addr_examples))):
            print(f"  BEFORE: {orig!r}")
            print(f"  AFTER : {norm!r}")

        # 5 examples of token extraction
        print(f"\n[Token Extraction] 5 Examples from {src}:")
        count = 0
        for addr in sample["business_address"]:
            if pd.isna(addr) or not addr or str(addr).strip().lower() == "null":
                continue
            digits = extract_digit_tokens(addr)
            words = extract_word_tokens(addr, min_len=4)
            print(f"  RAW ADDR : {addr!r}")
            print(f"  DIGITS (\\d{{3,}}): {sorted(list(digits))}")
            print(f"  WORD TOKENS (len>=4, ASCII): {sorted(list(words))}")
            count += 1
            if count >= 5:
                break



def run_blocking_evaluation(
    dataset_dir: str,
    name_cap: int = 2000,
    digit_cap: int = 5000,
    word_cap: int = 2000,
    sample_s1_limit: Optional[int] = None,
    target_country: Optional[str] = None
):
    """Steps 2, 3, and 4: Build indices, generate candidates, measure recall & distribution."""
    t_start = time.time()
    print("\n" + "=" * 80)
    print("STEPS 2, 3 & 4: INVERTED INDEX BLOCKING & RECALL BENCHMARK ON TRAIN")
    print("=" * 80)
    print(f"Configuration: name_cap={name_cap}, digit_cap={digit_cap}, word_cap={word_cap}")

    # 1. Load Ground Truth
    gt_path = os.path.join(dataset_dir, "train_ground_truth.tsv")
    print(f"\nLoading ground truth from {gt_path}...")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])

    def parse_ids(x):
        if not x or pd.isna(x):
            return []
        return [i.strip() for i in x.split(",") if i.strip()]

    gt_df["match_list"] = gt_df["matched_entity_ids"].apply(parse_ids)
    
    # Map s1_id -> list of matched IDs
    gt_map = dict(zip(gt_df["source1_entity_id"], gt_df["match_list"]))
    
    # Collect all matched S2/S3 IDs to track their script during indexing
    target_match_ids = set()
    for mlist in gt_df["match_list"]:
        for mid in mlist:
            target_match_ids.add(mid)

    del gt_df
    print(f"Loaded ground truth for {len(gt_map):,} S1 entities ({len(target_match_ids):,} unique matched target IDs).")

    # Discover country list from train_source1 or use target_country
    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    if target_country:
        countries = [c.strip() for c in target_country.split(",") if c.strip()]
        print(f"\nTarget country partition specified: {countries}")
    else:
        print("\nScanning countries from train_source1.tsv...")
        s1_countries = pd.read_csv(s1_path, sep="\t", usecols=["country"], dtype=str)["country"].unique()
        countries = sorted([c for c in s1_countries if c])
        print(f"Countries to process: {countries}")

    # Global tracking counters across all countries
    total_true_pairs = 0
    total_same_script_pairs = 0
    total_cross_script_pairs = 0

    # Recall hits counters
    all_hits_union = 0
    all_hits_name = 0
    all_hits_digit = 0
    all_hits_word = 0

    same_hits_union = 0
    same_hits_name = 0
    same_hits_digit = 0
    same_hits_word = 0

    cross_hits_union = 0
    cross_hits_name = 0
    cross_hits_digit = 0
    cross_hits_word = 0

    s1_cand_counts = []
    total_s1_processed = 0

    # Two-dimensional breakdown tracker: (country, script_group) -> metrics
    split_stats = defaultdict(lambda: {
        "s1_count": 0,
        "s1_with_gt": 0,
        "true_pairs": 0,
        "union_hits": 0,
        "name_hits": 0,
        "digit_hits": 0,
        "word_hits": 0,
        "cand_counts": []
    })

    # Non-Latin pass exclusivity tracking
    non_latin_exclusivity = Counter()

    # Store index capping statistics
    capping_reports = {}

    # Process each country partition independently
    for country in countries:
        t_c_start = time.time()
        print("\n" + "-" * 80)
        print(f"PROCESSING COUNTRY PARTITION: {country}")
        print("-" * 80)

        blocking_engine = CountryPartitionBlocking(
            country=country,
            name_cap=name_cap,
            digit_cap=digit_cap,
            word_cap=word_cap
        )

        match_script_map = {} # mid -> bool (True if latin)

        # Index train_source2 for this country
        s2_path = os.path.join(dataset_dir, "train_source2.tsv")
        print(f"Streaming and indexing {country} records from train_source2.tsv...")
        s2_count = 0
        for chunk in pd.read_csv(s2_path, sep="\t", dtype=str,
                                 usecols=["entity_id", "business_name", "business_address", "country"],
                                 keep_default_na=False, na_values=[""], chunksize=500000):
            c_chunk = chunk[chunk["country"] == country]
            if not c_chunk.empty:
                s2_count += len(c_chunk)
                blocking_engine.index_records(c_chunk)
                # Record script for targets
                for row in c_chunk[c_chunk["entity_id"].isin(target_match_ids)].itertuples(index=False):
                    match_script_map[row.entity_id] = is_latin_majority(row.business_name, threshold=0.5)

        # Index train_source3 for this country
        s3_path = os.path.join(dataset_dir, "train_source3.tsv")
        print(f"Streaming and indexing {country} records from train_source3.tsv...")
        s3_count = 0
        for chunk in pd.read_csv(s3_path, sep="\t", dtype=str,
                                 usecols=["entity_id", "business_name", "business_address", "country"],
                                 keep_default_na=False, na_values=[""], chunksize=500000):
            c_chunk = chunk[chunk["country"] == country]
            if not c_chunk.empty:
                s3_count += len(c_chunk)
                blocking_engine.index_records(c_chunk)
                # Record script for targets
                for row in c_chunk[c_chunk["entity_id"].isin(target_match_ids)].itertuples(index=False):
                    match_script_map[row.entity_id] = is_latin_majority(row.business_name, threshold=0.5)

        print(f"Indexed {s2_count:,} S2 records and {s3_count:,} S3 records for {country}.")

        # Finalize indices and print capping statistics
        print(f"Finalizing indices and applying frequency caps for {country}...")
        stats = blocking_engine.finalize()
        capping_reports[country] = stats

        print(f"\n--- {country} Index Posting Distribution & Frequency Capping ---")
        print(f"{'Index':<12} {'Total Keys':>12} {'Total Postings':>16} {'p50':>6} {'p90':>6} {'p99':>6} {'Max':>8} {'Cap':>6} {'Capped Keys':>14} {'Capped Postings %':>18}")
        print("-" * 110)
        for idx_name, s in stats.items():
            print(f"{idx_name:<12} {s['total_keys']:>12,d} {s['total_postings']:>16,d} {s['p50']:>6.0f} {s['p90']:>6.0f} {s['p99']:>6.0f} {s['max']:>8,d} {s['cap']:>6,d} {s['capped_keys']:>7,d} ({s['capped_keys_pct']:>4.2f}%) {s['capped_postings_pct']:>17.2f}%")

        # Query candidates for S1 records in this country
        print(f"\nStreaming S1 records for {country} and generating candidates...")
        c_s1_processed = 0
        for chunk in pd.read_csv(s1_path, sep="\t", dtype=str,
                                 usecols=["entity_id", "business_name", "business_address", "country"],
                                 keep_default_na=False, na_values=[""], chunksize=500000):
            c_chunk = chunk[chunk["country"] == country]
            if c_chunk.empty:
                continue

            for row in c_chunk.itertuples(index=False):
                s1_id = str(row.entity_id)
                s1_name = str(row.business_name) if pd.notna(row.business_name) else ""
                s1_addr = str(row.business_address) if pd.notna(row.business_address) else ""

                name_hits, digit_hits, word_hits, union_hits = blocking_engine.get_candidates_with_passes(
                    s1_name, s1_addr
                )
                cand_count = len(union_hits)
                s1_cand_counts.append(cand_count)
                c_s1_processed += 1
                total_s1_processed += 1

                # Script classification of S1 entity
                s1_is_latin = is_latin_majority(s1_name, threshold=0.5)
                s1_script = "Latin" if s1_is_latin else "Non-Latin"
                split_key = (country, s1_script)

                split_stats[split_key]["s1_count"] += 1
                split_stats[split_key]["cand_counts"].append(cand_count)

                # Ground truth evaluation
                true_matches = gt_map.get(s1_id, [])
                if true_matches:
                    split_stats[split_key]["s1_with_gt"] += 1
                    for mid in true_matches:
                        total_true_pairs += 1
                        split_stats[split_key]["true_pairs"] += 1

                        mid_is_latin = match_script_map.get(mid, True)
                        is_same_script = (s1_is_latin == mid_is_latin)

                        in_union = mid in union_hits
                        in_name = mid in name_hits
                        in_digit = mid in digit_hits
                        in_word = mid in word_hits

                        # All pairs
                        if in_union:
                            all_hits_union += 1
                            split_stats[split_key]["union_hits"] += 1
                        if in_name:
                            all_hits_name += 1
                            split_stats[split_key]["name_hits"] += 1
                        if in_digit:
                            all_hits_digit += 1
                            split_stats[split_key]["digit_hits"] += 1
                        if in_word:
                            all_hits_word += 1
                            split_stats[split_key]["word_hits"] += 1

                        # Non-Latin pass exclusivity tracking
                        if not s1_is_latin:
                            non_latin_exclusivity["total_pairs"] += 1
                            passes = []
                            if in_name: passes.append("Name")
                            if in_digit: passes.append("Digit")
                            if in_word: passes.append("Word")

                            if not passes:
                                non_latin_exclusivity["Missed"] += 1
                            else:
                                non_latin_exclusivity["Recalled"] += 1
                                key = " + ".join(passes)
                                non_latin_exclusivity[key] += 1

                        # Segment: Same script vs Cross script
                        if is_same_script:
                            total_same_script_pairs += 1
                            if in_union: same_hits_union += 1
                            if in_name: same_hits_name += 1
                            if in_digit: same_hits_digit += 1
                            if in_word: same_hits_word += 1
                        else:
                            total_cross_script_pairs += 1
                            if in_union: cross_hits_union += 1
                            if in_name: cross_hits_name += 1
                            if in_digit: cross_hits_digit += 1
                            if in_word: cross_hits_word += 1

                if sample_s1_limit and total_s1_processed >= sample_s1_limit:
                    break
            if sample_s1_limit and total_s1_processed >= sample_s1_limit:
                break

        t_c_elapsed = time.time() - t_c_start
        print(f"Finished {country} partition: {c_s1_processed:,} S1 records in {t_c_elapsed:.1f}s")
        del blocking_engine, match_script_map

    total_wall_clock = time.time() - t_start

    # --------------------------------------------------------------------------
    # STEP 4: FINAL RECALL & CANDIDATE DISTRIBUTION REPORT
    # --------------------------------------------------------------------------
    cand_counts_arr = np.array(s1_cand_counts, dtype=np.int32) if s1_cand_counts else np.array([0])
    total_candidate_pairs = int(cand_counts_arr.sum())

    rec_all = all_hits_union / total_true_pairs if total_true_pairs else 0.0
    rec_all_name = all_hits_name / total_true_pairs if total_true_pairs else 0.0
    rec_all_digit = all_hits_digit / total_true_pairs if total_true_pairs else 0.0
    rec_all_word = all_hits_word / total_true_pairs if total_true_pairs else 0.0

    rec_same = same_hits_union / total_same_script_pairs if total_same_script_pairs else 0.0
    rec_same_name = same_hits_name / total_same_script_pairs if total_same_script_pairs else 0.0
    rec_same_digit = same_hits_digit / total_same_script_pairs if total_same_script_pairs else 0.0
    rec_same_word = same_hits_word / total_same_script_pairs if total_same_script_pairs else 0.0

    rec_cross = cross_hits_union / total_cross_script_pairs if total_cross_script_pairs else 0.0
    rec_cross_name = cross_hits_name / total_cross_script_pairs if total_cross_script_pairs else 0.0
    rec_cross_digit = cross_hits_digit / total_cross_script_pairs if total_cross_script_pairs else 0.0
    rec_cross_word = cross_hits_word / total_cross_script_pairs if total_cross_script_pairs else 0.0

    p50 = float(np.percentile(cand_counts_arr, 50))
    p90 = float(np.percentile(cand_counts_arr, 90))
    p95 = float(np.percentile(cand_counts_arr, 95))
    p99 = float(np.percentile(cand_counts_arr, 99))
    c_max = int(cand_counts_arr.max())
    c_mean = float(cand_counts_arr.mean())

    print("\n" + "=" * 80)
    print("PHASE 2/3 COMPLETE SUMMARY TABLE")
    print("=" * 80)

    print("\n1. CANDIDATE RECALL BY SEGMENT & PASS BREAKDOWN:")
    print(f"{'Segment':<20} {'Total Pairs':>14} {'Union Recall':>14} {'Name Pass':>12} {'Digit Pass':>12} {'Word Pass':>12}")
    print("-" * 88)
    print(f"{'(a) All True Matches':<20} {total_true_pairs:>14,d} {rec_all*100:>13.2f}% {rec_all_name*100:>11.2f}% {rec_all_digit*100:>11.2f}% {rec_all_word*100:>11.2f}%")
    print(f"{'(b) Same-Script':<20} {total_same_script_pairs:>14,d} {rec_same*100:>13.2f}% {rec_same_name*100:>11.2f}% {rec_same_digit*100:>11.2f}% {rec_same_word*100:>11.2f}%")
    print(f"{'(c) Cross-Script':<20} {total_cross_script_pairs:>14,d} {rec_cross*100:>13.2f}% {rec_cross_name*100:>11.2f}% {rec_cross_digit*100:>11.2f}% {rec_cross_word*100:>11.2f}%")

    # 2. TWO-DIMENSIONAL BREAKDOWN (Country x S1 Script Group)
    print("\n2. TWO-DIMENSIONAL BREAKDOWN (COUNTRY x S1 SCRIPT GROUP):")
    hdr = f"{'Country':<10} {'S1 Script':<11} {'S1 Count':>10} {'w/ GT':>10} {'True Pairs':>12} {'Union Recall':>14} {'Name Pass':>11} {'Digit Pass':>12} {'Word Pass':>11} {'Mean Cands':>12} {'Median':>8} {'p90':>8} {'p99':>8} {'Max':>8}"
    print(hdr)
    print("-" * len(hdr))

    # Helper function to print a table row
    def print_split_row(label_c, label_s, s1_cnt, s1_gt, tp, u_hits, n_hits, d_hits, w_hits, c_list):
        u_rec = (u_hits / tp * 100) if tp else 0.0
        n_rec = (n_hits / tp * 100) if tp else 0.0
        d_rec = (d_hits / tp * 100) if tp else 0.0
        w_rec = (w_hits / tp * 100) if tp else 0.0
        c_arr = np.array(c_list, dtype=np.int32) if c_list else np.array([0])
        m_cand = float(c_arr.mean())
        p50_c = float(np.percentile(c_arr, 50))
        p90_c = float(np.percentile(c_arr, 90))
        p99_c = float(np.percentile(c_arr, 99))
        max_c = int(c_arr.max())
        print(f"{label_c:<10} {label_s:<11} {s1_cnt:>10,d} {s1_gt:>10,d} {tp:>12,d} {u_rec:>13.2f}% {n_rec:>10.2f}% {d_rec:>11.2f}% {w_rec:>10.2f}% {m_cand:>12.1f} {p50_c:>8.0f} {p90_c:>8.0f} {p99_c:>8.0f} {max_c:>8,d}")

    # Accumulators for overall script totals
    script_totals = defaultdict(lambda: {
        "s1_count": 0, "s1_with_gt": 0, "true_pairs": 0,
        "union_hits": 0, "name_hits": 0, "digit_hits": 0, "word_hits": 0,
        "cand_counts": []
    })

    for c in countries:
        c_s1 = 0
        c_gt = 0
        c_tp = 0
        c_uh = 0
        c_nh = 0
        c_dh = 0
        c_wh = 0
        c_cands = []

        for s_grp in ["Non-Latin", "Latin"]:
            sp = split_stats.get((c, s_grp))
            if sp:
                print_split_row(c, s_grp, sp["s1_count"], sp["s1_with_gt"], sp["true_pairs"],
                                sp["union_hits"], sp["name_hits"], sp["digit_hits"], sp["word_hits"],
                                sp["cand_counts"])
                c_s1 += sp["s1_count"]
                c_gt += sp["s1_with_gt"]
                c_tp += sp["true_pairs"]
                c_uh += sp["union_hits"]
                c_nh += sp["name_hits"]
                c_dh += sp["digit_hits"]
                c_wh += sp["word_hits"]
                c_cands.extend(sp["cand_counts"])

                # Add to script totals
                script_totals[s_grp]["s1_count"] += sp["s1_count"]
                script_totals[s_grp]["s1_with_gt"] += sp["s1_with_gt"]
                script_totals[s_grp]["true_pairs"] += sp["true_pairs"]
                script_totals[s_grp]["union_hits"] += sp["union_hits"]
                script_totals[s_grp]["name_hits"] += sp["name_hits"]
                script_totals[s_grp]["digit_hits"] += sp["digit_hits"]
                script_totals[s_grp]["word_hits"] += sp["word_hits"]
                script_totals[s_grp]["cand_counts"].extend(sp["cand_counts"])

        if len(countries) > 1:
            print_split_row(c, "Total", c_s1, c_gt, c_tp, c_uh, c_nh, c_dh, c_wh, c_cands)
            print("-" * len(hdr))

    # Overall summary if multiple countries
    if len(countries) > 1:
        print("-" * len(hdr))
        for s_grp in ["Non-Latin", "Latin"]:
            st = script_totals[s_grp]
            print_split_row("Overall", s_grp, st["s1_count"], st["s1_with_gt"], st["true_pairs"],
                            st["union_hits"], st["name_hits"], st["digit_hits"], st["word_hits"],
                            st["cand_counts"])
        print_split_row("Overall", "Total", total_s1_processed,
                        sum(st["s1_with_gt"] for st in script_totals.values()),
                        total_true_pairs, all_hits_union, all_hits_name, all_hits_digit, all_hits_word,
                        s1_cand_counts)
    print("-" * len(hdr))

    # 3. NON-LATIN PASS EXCLUSIVITY & ADDRESS WORD PASS CONTRIBUTION
    print("\n3. NON-LATIN SCRIPT: PASS EXCLUSIVITY & ADDRESS WORD CONTRIBUTION:")
    nl_total = non_latin_exclusivity["total_pairs"]
    nl_rec = non_latin_exclusivity["Recalled"]
    nl_miss = non_latin_exclusivity["Missed"]

    if nl_total > 0:
        print(f"  Total Non-Latin True Match Pairs : {nl_total:,d}")
        print(f"  Total Recalled (Union Hits)      : {nl_rec:,d} ({nl_rec/nl_total*100:.2f}%)")
        print(f"  Missed by All Passes             : {nl_miss:,d} ({nl_miss/nl_total*100:.2f}%)")
        print("\n  Pass Participation Combinations (Where Non-Latin matches were found):")
        combo_labels = [
            ("Name only (missed by Digit & Word)", "Name"),
            ("Digit only (missed by Name & Word)", "Digit"),
            ("Word only (UNIQUE to Word Pass)", "Word"),
            ("Name + Digit (redundant w/o Word)", "Name + Digit"),
            ("Name + Word", "Name + Word"),
            ("Digit + Word", "Digit + Word"),
            ("All Three (Name + Digit + Word)", "Name + Digit + Word")
        ]
        for desc, key in combo_labels:
            cnt = non_latin_exclusivity.get(key, 0)
            pct = (cnt / nl_total * 100) if nl_total else 0.0
            print(f"    - {desc:<38} : {cnt:>8,d} ({pct:>5.2f}%)")

        word_unique = non_latin_exclusivity.get("Word", 0)
        recalled_without_word = nl_rec - word_unique
        print("\n  Marginal Impact of Address Word-Token Pass on Non-Latin Recall:")
        print(f"    - Recall WITHOUT Address Word Pass (Name | Digit) : {recalled_without_word:,d} ({recalled_without_word/nl_total*100:.2f}%)")
        print(f"    - Recall WITH Address Word Pass (Full Union)      : {nl_rec:,d} ({nl_rec/nl_total*100:.2f}%)")
        print(f"    - Net Recall LOST if Address Word Pass removed     : {word_unique:,d} ({word_unique/nl_total*100:.2f}%)")
    else:
        print("  No Non-Latin true match pairs found in the evaluated partition.")

    print("\n4. CANDIDATE COUNT DISTRIBUTION PER S1 ENTITY:")
    print(f"  Total S1 Entities Processed: {total_s1_processed:,}")
    print(f"  Total Candidate Pairs Generated: {total_candidate_pairs:,}")
    print(f"  Mean Candidates per S1   : {c_mean:.2f}")
    print(f"  Median (p50) per S1      : {p50:.1f}")
    print(f"  p90 per S1               : {p90:.1f}")
    print(f"  p95 per S1               : {p95:.1f}")
    print(f"  p99 per S1               : {p99:.1f}")
    print(f"  Max Candidates per S1    : {c_max:,}")

    print("\n5. RUNTIME & SYSTEM PERFORMANCE:")
    print(f"  Total Wall-Clock Runtime : {total_wall_clock:.1f} seconds ({total_wall_clock/60:.2f} minutes)")
    print(f"  Processing Speed         : {total_s1_processed/total_wall_clock:,.1f} S1 records/second")

    print("\n6. TARGET RECALL CHECK (Target >= 0.97):")
    if rec_all >= 0.97:
        print(f"  >> SUCCESS: Overall Recall = {rec_all*100:.3f}% >= 97.000%")
    else:
        print(f"  >> NOTICE: Overall Recall = {rec_all*100:.3f}% (Expected ceiling due to ~4% non-numeric cross-script: ~{100 - (total_cross_script_pairs - cross_hits_union)/total_true_pairs*100:.2f}%)")

    print("\n" + "=" * 80)
    print("PHASE 2/3 COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Phase 2/3 Normalization & Blocking Evaluation")
    parser.add_argument("--dataset-dir", default=os.path.join(BASE_DIR, "dataset", "train"),
                        help="Path to train dataset directory")
    parser.add_argument("--name-cap", type=int, default=2000, help="Name token posting list cap")
    parser.add_argument("--digit-cap", type=int, default=5000, help="Digit token posting list cap")
    parser.add_argument("--word-cap", type=int, default=2000, help="Address Latin word posting list cap")
    parser.add_argument("--sample", type=int, default=None, help="Sample N S1 records for quick test")
    parser.add_argument("--step1-only", action="store_true", help="Run Step 1 normalization check only")
    parser.add_argument("--country", "--countries", dest="country", type=str, default=None,
                        help="Process only specific country partition(s) (e.g. India or India,US)")
    args = parser.parse_args()

    sys.stdout = Logger(OUTPUT_LOG)

    # 1. Normalization sanity check
    run_step1_normalization_check(args.dataset_dir)

    if args.step1_only:
        print("\n" + "=" * 80)
        print("STEP 1 COMPLETE (--step1-only specified, stopping before index building)")
        print("=" * 80)
        return

    # 2, 3 & 4. Blocking index build & recall measurement
    run_blocking_evaluation(
        dataset_dir=args.dataset_dir,
        name_cap=args.name_cap,
        digit_cap=args.digit_cap,
        word_cap=args.word_cap,
        sample_s1_limit=args.sample,
        target_country=args.country
    )

if __name__ == "__main__":
    main()
