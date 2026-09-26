"""
Phase 1b Ground-Truth Analysis Script for Business Entity Resolution.
Computes:
  1. Country-blocking safety: exact cross-country mismatch rate among ground-truth matches
  2. True-match pair script classification (latin_majority vs nonlatin_majority).
  3. Counts for:
     - S1=latin, match=latin
     - S1=latin, match=nonlatin (cross-script risk case)
     - Breakdown by country and match source (S2 vs S3)
  4. Digit sequence (\\d{3,}) overlap in business_address for cross-script pairs.
  5. Test set candidate volume estimate if blocked purely by country.

Saves full output to phase1b_output.txt and prints to console.
All results printed as plain numbers with no interpretation.
"""

import os
import sys
import re
import unicodedata
from collections import Counter
import pandas as pd
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_LOG = os.path.join(BASE_DIR, "phase1b_output.txt")

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

_CHAR_CACHE = {}

def latin_fraction(s):
    if not isinstance(s, str) or not s:
        return 1.0
    total = 0
    latin = 0
    for ch in s:
        info = _CHAR_CACHE.get(ch)
        if info is None:
            if ch.isspace() or unicodedata.category(ch).startswith("P"):
                info = (True, False)
            else:
                try:
                    name = unicodedata.name(ch)
                    info = (False, "LATIN" in name)
                except ValueError:
                    info = (False, False)
            _CHAR_CACHE[ch] = info
        
        skip, is_lat = info
        if skip:
            continue
        total += 1
        if is_lat:
            latin += 1
            
    return latin / total if total else 1.0

DIGIT_RE = re.compile(r'\d{3,}')

def extract_digit_tokens(address):
    if not address or not isinstance(address, str):
        return set()
    return set(DIGIT_RE.findall(address))

def main():
    sys.stdout = Logger(OUTPUT_LOG)
    t0 = time.time()

    print("================================================================================")
    print("PHASE 1B: EXTENDED GROUND-TRUTH SCRIPT & NUMERIC ADDRESS ANALYSIS")
    print("================================================================================")

    # 1. Load Ground Truth
    gt_path = os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv")
    print(f"Loading ground truth: {gt_path} ...")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])

    def parse_ids(x):
        if pd.isna(x) or str(x).strip() == "":
            return []
        return [i.strip() for i in str(x).split(",") if i.strip()]

    gt["match_list"] = gt["matched_entity_ids"].apply(parse_ids)
    
    # Collect all (s1_id, match_id) pairs
    s1_ids = []
    match_ids = []
    target_s2_ids = set()
    target_s3_ids = set()
    s1_needed_ids = set()

    for s1_id, mlist in zip(gt["source1_entity_id"], gt["match_list"]):
        if not mlist:
            continue
        s1_needed_ids.add(s1_id)
        for mid in mlist:
            s1_ids.append(s1_id)
            match_ids.append(mid)
            if mid.startswith("S2-"):
                target_s2_ids.add(mid)
            elif mid.startswith("S3-"):
                target_s3_ids.add(mid)

    total_pairs = len(s1_ids)
    print(f"Total S1 entities with matches: {len(s1_needed_ids):,}")
    print(f"Total true-match pairs: {total_pairs:,}")
    print(f"Unique matched S2 records: {len(target_s2_ids):,}")
    print(f"Unique matched S3 records: {len(target_s3_ids):,}")

    # 2. Load S1 data
    print("Loading train_source1.tsv ...")
    s1_path = os.path.join(DATASET_DIR, "train", "train_source1.tsv")
    s1_data = {} # id -> (name, address, country)
    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str,
                             usecols=["entity_id", "business_name", "business_address", "country"],
                             keep_default_na=False, na_values=[""], chunksize=500000):
        chunk_needed = chunk[chunk["entity_id"].isin(s1_needed_ids)]
        for row in chunk_needed.itertuples(index=False):
            s1_data[row.entity_id] = (row.business_name, row.business_address, row.country)
    print(f"Loaded {len(s1_data):,} S1 records.")

    # 3. Load matched S2 and S3 data
    match_data = {} # id -> (name, address, country)
    print("Loading matched records from train_source2.tsv ...")
    s2_path = os.path.join(DATASET_DIR, "train", "train_source2.tsv")
    for chunk in pd.read_csv(s2_path, sep="\t", dtype=str,
                             usecols=["entity_id", "business_name", "business_address", "country"],
                             keep_default_na=False, na_values=[""], chunksize=500000):
        chunk_needed = chunk[chunk["entity_id"].isin(target_s2_ids)]
        for row in chunk_needed.itertuples(index=False):
            match_data[row.entity_id] = (row.business_name, row.business_address, row.country)

    print("Loading matched records from train_source3.tsv ...")
    s3_path = os.path.join(DATASET_DIR, "train", "train_source3.tsv")
    for chunk in pd.read_csv(s3_path, sep="\t", dtype=str,
                             usecols=["entity_id", "business_name", "business_address", "country"],
                             keep_default_na=False, na_values=[""], chunksize=500000):
        chunk_needed = chunk[chunk["entity_id"].isin(target_s3_ids)]
        for row in chunk_needed.itertuples(index=False):
            match_data[row.entity_id] = (row.business_name, row.business_address, row.country)

    print(f"Loaded {len(match_data):,} matched S2/S3 records.")

    # 4. Classify script and examine pairs
    print("\nProcessing true-match pairs...")
    
    country_mismatches = 0
    country_mismatch_examples = []

    pair_script_counts = Counter()
    latin_nonlatin_by_country = Counter()
    latin_nonlatin_by_source = Counter()
    latin_nonlatin_by_country_and_source = Counter()

    cross_script_pairs_total = 0
    cross_script_digit_overlap_count = 0
    cross_script_s1_has_digits_count = 0
    cross_script_match_has_digits_count = 0
    cross_script_both_have_digits_count = 0

    all_cross_script_total = 0
    all_cross_digit_overlap_count = 0

    missing_pairs = 0
    name_class_cache = {}

    def get_class(name):
        res = name_class_cache.get(name)
        if res is None:
            frac = latin_fraction(name)
            res = "latin_majority" if frac >= 0.5 else "nonlatin_majority"
            name_class_cache[name] = res
        return res

    for s1_id, mid in zip(s1_ids, match_ids):
        s1_info = s1_data.get(s1_id)
        mid_info = match_data.get(mid)
        if not s1_info or not mid_info:
            missing_pairs += 1
            continue

        s1_name, s1_addr, s1_country = s1_info
        mid_name, mid_addr, mid_country = mid_info

        # Check cross-country
        if s1_country != mid_country:
            country_mismatches += 1
            if len(country_mismatch_examples) < 10:
                country_mismatch_examples.append((s1_id, s1_country, mid, mid_country))

        s1_class = get_class(s1_name)
        mid_class = get_class(mid_name)

        pair_script_counts[(s1_class, mid_class)] += 1

        match_source = "S2" if mid.startswith("S2-") else "S3"
        pair_country = s1_country

        if s1_class != mid_class:
            all_cross_script_total += 1
            s1_digits = extract_digit_tokens(s1_addr)
            mid_digits = extract_digit_tokens(mid_addr)
            if bool(s1_digits & mid_digits):
                all_cross_digit_overlap_count += 1

        if s1_class == "latin_majority" and mid_class == "nonlatin_majority":
            cross_script_pairs_total += 1
            latin_nonlatin_by_country[pair_country] += 1
            latin_nonlatin_by_source[match_source] += 1
            latin_nonlatin_by_country_and_source[(pair_country, match_source)] += 1

            s1_digits = extract_digit_tokens(s1_addr)
            mid_digits = extract_digit_tokens(mid_addr)

            if s1_digits:
                cross_script_s1_has_digits_count += 1
            if mid_digits:
                cross_script_match_has_digits_count += 1
            if s1_digits and mid_digits:
                cross_script_both_have_digits_count += 1
            if bool(s1_digits & mid_digits):
                cross_script_digit_overlap_count += 1

    # --------------------------------------------------------------------------
    # PRINT RESULTS AS PLAIN NUMBERS
    # --------------------------------------------------------------------------
    evaluated_pairs = total_pairs - missing_pairs

    print("\n" + "=" * 80)
    print("RESULTS (PLAIN NUMBERS)")
    print("=" * 80)

    print(f"\n1. COUNTRY MISMATCH STATS")
    print(f"total_ground_truth_pairs: {total_pairs}")
    print(f"evaluated_pairs: {evaluated_pairs}")
    print(f"country_mismatch_count: {country_mismatches}")
    print(f"country_mismatch_rate: {country_mismatches / evaluated_pairs if evaluated_pairs else 0:.8f}")

    print(f"\n2. SCRIPT CLASSIFICATION COUNTS (latin_fraction threshold 0.5)")
    n_lat_lat = pair_script_counts[("latin_majority", "latin_majority")]
    n_lat_nonlat = pair_script_counts[("latin_majority", "nonlatin_majority")]
    n_nonlat_lat = pair_script_counts[("nonlatin_majority", "latin_majority")]
    n_nonlat_nonlat = pair_script_counts[("nonlatin_majority", "nonlatin_majority")]

    print(f"s1_latin_match_latin: {n_lat_lat}")
    print(f"s1_latin_match_latin_fraction: {n_lat_lat / evaluated_pairs if evaluated_pairs else 0:.8f}")
    print(f"s1_latin_match_nonlatin: {n_lat_nonlat}")
    print(f"s1_latin_match_nonlatin_fraction: {n_lat_nonlat / evaluated_pairs if evaluated_pairs else 0:.8f}")
    print(f"s1_nonlatin_match_latin: {n_nonlat_lat}")
    print(f"s1_nonlatin_match_latin_fraction: {n_nonlat_lat / evaluated_pairs if evaluated_pairs else 0:.8f}")
    print(f"s1_nonlatin_match_nonlatin: {n_nonlat_nonlat}")
    print(f"s1_nonlatin_match_nonlatin_fraction: {n_nonlat_nonlat / evaluated_pairs if evaluated_pairs else 0:.8f}")
    print(f"all_cross_script_count: {all_cross_script_total}")
    print(f"all_cross_script_fraction: {all_cross_script_total / evaluated_pairs if evaluated_pairs else 0:.8f}")

    print(f"\n3. S1=latin, match=nonlatin BREAKDOWN BY COUNTRY")
    for country, count in sorted(latin_nonlatin_by_country.items()):
        frac = count / n_lat_nonlat if n_lat_nonlat > 0 else 0.0
        print(f"country_{country}_count: {count}")
        print(f"country_{country}_fraction_of_s1_latin_match_nonlatin: {frac:.8f}")

    print(f"\n4. S1=latin, match=nonlatin BREAKDOWN BY MATCH SOURCE")
    for source, count in sorted(latin_nonlatin_by_source.items()):
        frac = count / n_lat_nonlat if n_lat_nonlat > 0 else 0.0
        print(f"source_{source}_count: {count}")
        print(f"source_{source}_fraction_of_s1_latin_match_nonlatin: {frac:.8f}")

    print(f"\n5. S1=latin, match=nonlatin BREAKDOWN BY COUNTRY AND SOURCE")
    for (country, source), count in sorted(latin_nonlatin_by_country_and_source.items()):
        frac = count / n_lat_nonlat if n_lat_nonlat > 0 else 0.0
        print(f"country_{country}_source_{source}_count: {count}")
        print(f"country_{country}_source_{source}_fraction_of_s1_latin_match_nonlatin: {frac:.8f}")

    print(f"\n6. NUMERIC ADDRESS DIGIT OVERLAP (regex \\d{{3,}}) FOR S1=latin, match=nonlatin")
    overlap_frac = cross_script_digit_overlap_count / cross_script_pairs_total if cross_script_pairs_total > 0 else 0.0
    print(f"cross_script_pairs_evaluated: {cross_script_pairs_total}")
    print(f"cross_script_shared_digit_overlap_count: {cross_script_digit_overlap_count}")
    print(f"cross_script_shared_digit_overlap_fraction: {overlap_frac:.8f}")
    print(f"cross_script_s1_has_digits_count: {cross_script_s1_has_digits_count}")
    print(f"cross_script_s1_has_digits_fraction: {cross_script_s1_has_digits_count / cross_script_pairs_total if cross_script_pairs_total else 0:.8f}")
    print(f"cross_script_match_has_digits_count: {cross_script_match_has_digits_count}")
    print(f"cross_script_match_has_digits_fraction: {cross_script_match_has_digits_count / cross_script_pairs_total if cross_script_pairs_total else 0:.8f}")
    print(f"cross_script_both_have_digits_count: {cross_script_both_have_digits_count}")
    print(f"cross_script_both_have_digits_fraction: {cross_script_both_have_digits_count / cross_script_pairs_total if cross_script_pairs_total else 0:.8f}")

    # Free memory
    del s1_data, match_data, gt

    # 7. Test Set pure country blocking volume estimate
    print(f"\n7. TEST SET CANDIDATE VOLUME ESTIMATE (PURE COUNTRY BLOCKING)")
    ts1_c = pd.read_csv(os.path.join(DATASET_DIR, "test", "test_source1.tsv"), sep="\t", usecols=["country"], dtype=str)["country"].value_counts()
    ts2_c = pd.read_csv(os.path.join(DATASET_DIR, "test", "test_source2.tsv"), sep="\t", usecols=["country"], dtype=str)["country"].value_counts()
    ts3_c = pd.read_csv(os.path.join(DATASET_DIR, "test", "test_source3.tsv"), sep="\t", usecols=["country"], dtype=str)["country"].value_counts()

    all_countries = sorted(list(set(ts1_c.index) | set(ts2_c.index) | set(ts3_c.index)))
    total_test_pairs = 0
    for c in all_countries:
        n1 = int(ts1_c.get(c, 0))
        n2 = int(ts2_c.get(c, 0))
        n3 = int(ts3_c.get(c, 0))
        pairs = n1 * (n2 + n3)
        total_test_pairs += pairs
        print(f"test_country_{c}_s1_count: {n1}")
        print(f"test_country_{c}_s2_count: {n2}")
        print(f"test_country_{c}_s3_count: {n3}")
        print(f"test_country_{c}_s2_u_s3_count: {n2 + n3}")
        print(f"test_country_{c}_candidate_pairs: {pairs}")

    print(f"total_test_candidate_pairs_pure_country: {total_test_pairs}")

    elapsed = time.time() - t0
    print(f"\ntotal_runtime_seconds: {elapsed:.2f}")
    print("================================================================================")
    print("PHASE 1B COMPLETE")
    print("================================================================================")

if __name__ == "__main__":
    main()
