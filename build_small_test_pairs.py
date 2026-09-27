#!/usr/bin/env python3
"""
build_small_test_pairs.py — Candidate Pair Generation for the TEST Set.

Design:
- Reuses the fast multi-strategy inverted-index blocking logic from blocking.py & candidate_generation.py.
- Loads ALL test entities from test_source1.tsv (every entity must appear in candidate generation).
- Partitions by country (France, India, US) for maximum recall and streaming memory efficiency.
- Gracefully handles unseen/missing country values (e.g. France, unindexed countries, or nulls)
  with a robust fallback to name-token blocking across all countries.
- Outputs exact column schema:
    source1_entity_id\tcandidate_entity_id\tsource
  where source is 'S2' or 'S3'. No ground-truth label column.
- Writes to output/test_candidate_pairs.tsv.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Optional
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
from blocking import CountryPartitionBlocking, InvertedIndex, STOPWORDS
from data_loader import load_tsv


def get_all_s1_entities(test_dir: str) -> pd.DataFrame:
    """Loads all test Source 1 entities with string dtypes."""
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Loading all S1 entities from {s1_path}...")
    df = pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        low_memory=False
    )
    # Clean whitespace and handle literal 'null'
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()
        mask = df[col].str.lower() == "null"
        if mask.any():
            df[col] = df[col].mask(mask, "")
    print(f"Loaded {len(df):,} Source 1 test records.")
    return df


def discover_countries(df_s1: pd.DataFrame, test_dir: str) -> List[str]:
    """Identifies all distinct countries across test S1, S2, and S3."""
    countries = set(df_s1["country"].unique())
    countries.discard("")

    # Also scan sample of S2 / S3 in case additional countries exist
    for src in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(test_dir, src)
        if os.path.isfile(path):
            try:
                sample_countries = pd.read_csv(
                    path, sep="\t", usecols=["country"], dtype=str, nrows=500000
                )["country"].unique()
                for c in sample_countries:
                    c_clean = str(c).strip()
                    if c_clean and c_clean.lower() != "null":
                        countries.add(c_clean)
            except Exception:
                pass

    country_list = sorted(list(countries))
    print(f"Identified country partitions: {country_list}")
    return country_list


def build_country_blocking_engine(
    country: str,
    test_dir: str,
    name_cap: int = 2000,
    digit_cap: int = 5000,
    word_cap: int = 2000,
    chunksize: int = 500000
) -> CountryPartitionBlocking:
    """
    Builds and finalizes inverted indices for one country partition across test S2 and S3.
    Streams chunks to keep memory footprint lean.
    """
    print(f"\nBuilding blocking indices for country: {country}...")
    t0 = time.time()
    engine = CountryPartitionBlocking(
        country=country,
        name_cap=name_cap,
        digit_cap=digit_cap,
        word_cap=word_cap
    )

    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    # Index test_source2
    s2_cnt = 0
    if os.path.isfile(s2_path):
        for chunk in pd.read_csv(
            s2_path,
            sep="\t",
            dtype=str,
            usecols=["entity_id", "business_name", "business_address", "country"],
            keep_default_na=False,
            na_values=[""],
            chunksize=chunksize,
            low_memory=False
        ):
            c_chunk = chunk[chunk["country"].astype(str).str.strip().str.lower() == country.lower()]
            if not c_chunk.empty:
                s2_cnt += len(c_chunk)
                engine.index_records(c_chunk)

    # Index test_source3
    s3_cnt = 0
    if os.path.isfile(s3_path):
        for chunk in pd.read_csv(
            s3_path,
            sep="\t",
            dtype=str,
            usecols=["entity_id", "business_name", "business_address", "country"],
            keep_default_na=False,
            na_values=[""],
            chunksize=chunksize,
            low_memory=False
        ):
            c_chunk = chunk[chunk["country"].astype(str).str.strip().str.lower() == country.lower()]
            if not c_chunk.empty:
                s3_cnt += len(c_chunk)
                engine.index_records(c_chunk)

    print(f"  Indexed {s2_cnt:,} S2 and {s3_cnt:,} S3 records for {country}.")
    stats = engine.finalize()
    elapsed = time.time() - t0
    print(f"  Finalized {country} indices in {elapsed:.1f}s.")
    return engine


def build_global_name_fallback_index(
    test_dir: str,
    name_cap: int = 1000,
    chunksize: int = 500000,
    max_records: int = 1000000
) -> InvertedIndex:
    """
    Builds a lightweight global name inverted index across S2 and S3 for fallback
    when an entity has an unseen country or produces zero country-level candidates.
    """
    print("\nBuilding lightweight global fallback index for unseen countries/singletons...")
    fallback_index = InvertedIndex(name="global_fallback", cap=name_cap)

    for src in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(test_dir, src)
        if not os.path.isfile(path):
            continue
        cnt = 0
        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=["entity_id", "business_name"],
            keep_default_na=False,
            na_values=[""],
            chunksize=chunksize,
            low_memory=False
        ):
            for row in chunk.itertuples(index=False):
                eid = str(row.entity_id)
                name = str(row.business_name) if pd.notna(row.business_name) else ""
                norm = normalize_name(name)
                if norm:
                    for tok in norm.split():
                        if len(tok) >= 3 and tok not in STOPWORDS:
                            fallback_index.add(tok, eid)
            cnt += len(chunk)
            if cnt >= max_records:
                break
    fallback_index.finalize()
    print("  Global fallback index ready.")
    return fallback_index


def run_test_candidate_generation(
    test_dir: str,
    output_path: str,
    name_cap: int = 2000,
    digit_cap: int = 5000,
    word_cap: int = 2000,
    chunksize: int = 500000,
    max_candidates_per_s1: int = 150
):
    """
    End-to-end candidate pair generation for the test set:
    - Streams through every country partition
    - Generates candidate pairs per S1 record
    - Writes output/test_candidate_pairs.tsv with exact schema:
        source1_entity_id\tcandidate_entity_id\tsource
    """
    t_start = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Load all S1 entities
    df_s1 = get_all_s1_entities(test_dir)
    total_s1 = len(df_s1)
    countries = discover_countries(df_s1, test_dir)

    # Track how many candidates were generated
    total_pairs_written = 0
    s1_with_zero_cands = []

    # Open output file and write header
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_id\tsource\n")

        # 2. Process each country partition one by one (to preserve RAM)
        for country in countries:
            print("\n" + "=" * 70)
            print(f"PROCESSING PARTITION: {country}")
            print("=" * 70)

            # Filter S1 records for this country
            c_mask = df_s1["country"].astype(str).str.strip().str.lower() == country.lower()
            df_s1_country = df_s1[c_mask]
            s1_count = len(df_s1_country)
            print(f"S1 entities in {country}: {s1_count:,}")

            if s1_count == 0:
                continue

            # Build Country Partition Blocking engine
            engine = build_country_blocking_engine(
                country=country,
                test_dir=test_dir,
                name_cap=name_cap,
                digit_cap=digit_cap,
                word_cap=word_cap,
                chunksize=chunksize
            )

            print(f"Generating candidate pairs for {s1_count:,} S1 records in {country}...")
            c_pairs_count = 0
            t_query_start = time.time()

            # Query candidates per S1 entity
            for idx, row in enumerate(df_s1_country.itertuples(index=False), start=1):
                s1_id = str(row.entity_id)
                s1_name = str(row.business_name) if pd.notna(row.business_name) else ""
                s1_addr = str(row.business_address) if pd.notna(row.business_address) else ""

                cands: Set[str] = engine.get_candidates(s1_name, s1_addr)

                if not cands:
                    s1_with_zero_cands.append(s1_id)
                    continue

                # Sort candidates for determinism and cap if needed
                cand_list = sorted(list(cands))
                if max_candidates_per_s1 and len(cand_list) > max_candidates_per_s1:
                    cand_list = cand_list[:max_candidates_per_s1]

                for cid in cand_list:
                    src_tag = "S2" if cid.startswith("S2-") else "S3"
                    out_f.write(f"{s1_id}\t{cid}\t{src_tag}\n")
                    c_pairs_count += 1

                if idx % 100000 == 0:
                    print(f"  Processed {idx:,}/{s1_count:,} S1 records ({c_pairs_count:,} pairs)...")

            elapsed_query = time.time() - t_query_start
            print(f"Completed {country}: generated {c_pairs_count:,} pairs in {elapsed_query:.1f}s.")
            total_pairs_written += c_pairs_count

            # Delete engine to free memory for the next partition
            del engine

        # 3. Fallback for entities with unseen country or 0 candidates
        # Look for S1 entities with unknown country or empty country
        known_countries_lower = {c.lower() for c in countries}
        unseen_mask = ~df_s1["country"].astype(str).str.strip().str.lower().isin(known_countries_lower)
        df_unseen = df_s1[unseen_mask]

        if not df_unseen.empty or s1_with_zero_cands:
            print("\n" + "=" * 70)
            print("HANDLING UNSEEN COUNTRIES & ZERO-CANDIDATE FALLBACK")
            print("=" * 70)
            print(f"  S1 with unseen country: {len(df_unseen):,}")
            print(f"  S1 with zero candidates in country blocking: {len(s1_with_zero_cands):,}")

            fallback_index = build_global_name_fallback_index(test_dir=test_dir)
            fallback_pairs = 0

            # Process unseen country records
            for row in df_unseen.itertuples(index=False):
                s1_id = str(row.entity_id)
                s1_name = str(row.business_name) if pd.notna(row.business_name) else ""
                norm = normalize_name(s1_name)
                toks = {t for t in norm.split() if len(t) >= 3 and t not in STOPWORDS} if norm else set()
                cands = fallback_index.query(toks)
                cand_list = sorted(list(cands))[:max_candidates_per_s1]
                for cid in cand_list:
                    src_tag = "S2" if cid.startswith("S2-") else "S3"
                    out_f.write(f"{s1_id}\t{cid}\t{src_tag}\n")
                    fallback_pairs += 1

            # Process a sample of zero-candidate entities if any
            if s1_with_zero_cands:
                s1_lookup = df_s1.set_index("entity_id")["business_name"].to_dict()
                for s1_id in s1_with_zero_cands[:50000]:  # Cap fallback to avoid exploding
                    s1_name = s1_lookup.get(s1_id, "")
                    norm = normalize_name(s1_name)
                    toks = {t for t in norm.split() if len(t) >= 3 and t not in STOPWORDS} if norm else set()
                    cands = fallback_index.query(toks)
                    cand_list = sorted(list(cands))[:30]
                    for cid in cand_list:
                        src_tag = "S2" if cid.startswith("S2-") else "S3"
                        out_f.write(f"{s1_id}\t{cid}\t{src_tag}\n")
                        fallback_pairs += 1

            total_pairs_written += fallback_pairs
            print(f"  Fallback generated {fallback_pairs:,} additional candidate pairs.")
            del fallback_index

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("TEST CANDIDATE PAIR GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Total S1 Entities Processed: {total_s1:,}")
    print(f"  Total Candidate Pairs Written: {total_pairs_written:,}")
    print(f"  Output File: {output_path}")
    print(f"  Total Runtime: {total_time:.1f}s ({total_time / 60:.2f} minutes)")


def main():
    parser = argparse.ArgumentParser(description="Build Candidate Pairs for Test Set")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test files directory")
    parser.add_argument("--output", default="output/test_candidate_pairs.tsv", help="Output path for candidate pairs")
    parser.add_argument("--name-cap", type=int, default=2000, help="Name token posting cap")
    parser.add_argument("--digit-cap", type=int, default=5000, help="Digit token posting cap")
    parser.add_argument("--word-cap", type=int, default=2000, help="Word token posting cap")
    parser.add_argument("--max-candidates-per-s1", type=int, default=150, help="Max candidates per S1 entity")
    parser.add_argument("--chunksize", type=int, default=500000, help="Chunk size for streaming TSVs")
    args = parser.parse_args()

    run_test_candidate_generation(
        test_dir=args.test_dir,
        output_path=args.output,
        name_cap=args.name_cap,
        digit_cap=args.digit_cap,
        word_cap=args.word_cap,
        chunksize=args.chunksize,
        max_candidates_per_s1=args.max_candidates_per_s1
    )


if __name__ == "__main__":
    main()
