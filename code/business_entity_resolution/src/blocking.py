"""
Inverted-index blocking module for Business Entity Resolution pipeline.
Implements:
  - Multi-strategy inverted indices (name tokens, address digits, address Latin words)
  - Frequency capping per index
  - Per-country partitioned candidate generation
  - Instrumented recall, size, and runtime measurement on train
"""

import os
import sys
import time
import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, Set, List, Tuple, Optional

# Ensure imports work whether run from root or package
try:
    from .normalization import (
        normalize_name,
        normalize_address,
        extract_digit_tokens,
        extract_word_tokens,
        clean_text
    )
    from .data_loader import load_tsv, parse_ground_truth
except ImportError:
    from normalization import (
        normalize_name,
        normalize_address,
        extract_digit_tokens,
        extract_word_tokens,
        clean_text
    )
    from data_loader import load_tsv, parse_ground_truth

STOPWORDS = {"and", "the", "of", "in", "for", "at", "to", "a", "an", "on", "by"}

class InvertedIndex:
    """Inverted index mapping token -> set of entity IDs with frequency capping."""
    def __init__(self, name: str, cap: int):
        self.name = name
        self.cap = cap
        self.index: Dict[str, Set[str]] = defaultdict(set)
        self.capped_keys: Set[str] = set()
        self.active_index: Dict[str, Set[str]] = {}

    def add(self, key: str, entity_id: str):
        if key and len(key) >= 2 and key not in STOPWORDS:
            self.index[key].add(entity_id)

    def finalize(self) -> Dict[str, float]:
        """Calculates posting distributions, applies frequency capping, and activates index."""
        postings_sizes = [len(v) for v in self.index.values()]
        if not postings_sizes:
            return {"count": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "capped_keys": 0, "capped_postings": 0}

        sizes = np.array(postings_sizes)
        total_keys = len(sizes)
        total_postings = int(sizes.sum())

        self.capped_keys = {k for k, v in self.index.items() if len(v) > self.cap}
        self.active_index = {k: v for k, v in self.index.items() if len(v) <= self.cap}

        capped_postings = sum(len(self.index[k]) for k in self.capped_keys)

        stats = {
            "total_keys": total_keys,
            "total_postings": total_postings,
            "p50": float(np.percentile(sizes, 50)),
            "p90": float(np.percentile(sizes, 90)),
            "p99": float(np.percentile(sizes, 99)),
            "max": int(sizes.max()),
            "cap": self.cap,
            "capped_keys": len(self.capped_keys),
            "capped_keys_pct": len(self.capped_keys) / total_keys * 100 if total_keys else 0.0,
            "capped_postings": capped_postings,
            "capped_postings_pct": capped_postings / total_postings * 100 if total_postings else 0.0,
            "active_keys": len(self.active_index)
        }
        # Clear raw index to conserve memory
        self.index.clear()
        return stats

    def query(self, tokens: Set[str]) -> Set[str]:
        hits: Set[str] = set()
        for tok in tokens:
            postings = self.active_index.get(tok)
            if postings:
                hits.update(postings)
        return hits


class CountryPartitionBlocking:
    """Manages inverted indices for S2 and S3 within a single country partition."""
    def __init__(self, country: str, name_cap: int = 2000, digit_cap: int = 5000, word_cap: int = 2000):
        self.country = country
        self.name_cap = name_cap
        self.digit_cap = digit_cap
        self.word_cap = word_cap

        # Combined indices across S2 and S3 for this country
        self.name_index = InvertedIndex(f"{country}_name", cap=name_cap)
        self.digit_index = InvertedIndex(f"{country}_digit", cap=digit_cap)
        self.word_index = InvertedIndex(f"{country}_word", cap=word_cap)

    def index_records(self, df_records: pd.DataFrame):
        """Builds indices from a DataFrame containing entity_id, business_name, business_address."""
        for row in df_records.itertuples(index=False):
            eid = str(row.entity_id)
            name = str(row.business_name) if pd.notna(row.business_name) else ""
            addr = str(row.business_address) if pd.notna(row.business_address) else ""

            # 1. Normalized name tokens (length >= 2)
            norm_name = normalize_name(name)
            if norm_name:
                for tok in norm_name.split():
                    if len(tok) >= 2:
                        self.name_index.add(tok, eid)

            # 2. Address digit tokens (regex \d{3,})
            digits = extract_digit_tokens(addr)
            for d in digits:
                self.digit_index.add(d, eid)

            # 3. Address Latin word tokens (length >= 4)
            words = extract_word_tokens(addr, min_len=4)
            for w in words:
                self.word_index.add(w, eid)

    def finalize(self) -> Dict[str, Dict[str, float]]:
        s_name = self.name_index.finalize()
        s_digit = self.digit_index.finalize()
        s_word = self.word_index.finalize()
        return {
            "name": s_name,
            "digit": s_digit,
            "word": s_word
        }

    def get_candidates_with_passes(
        self,
        name: str,
        address: str
    ) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
        """
        Returns (name_hits, digit_hits, word_hits, union_hits).
        """
        # 1. Name tokens
        norm_name = normalize_name(name)
        name_toks = {t for t in norm_name.split() if len(t) >= 2 and t not in STOPWORDS} if norm_name else set()
        name_hits = self.name_index.query(name_toks)

        # 2. Digit tokens
        digit_toks = extract_digit_tokens(address)
        digit_hits = self.digit_index.query(digit_toks)

        # 3. Word tokens
        word_toks = extract_word_tokens(address, min_len=4)
        word_hits = self.word_index.query(word_toks)

        # Union
        union_hits = name_hits | digit_hits | word_hits
        return name_hits, digit_hits, word_hits, union_hits

    def get_candidates(self, name: str, address: str) -> Set[str]:
        """Returns union candidate set directly for speed."""
        norm_name = normalize_name(name)
        name_toks = {t for t in norm_name.split() if len(t) >= 2 and t not in STOPWORDS} if norm_name else set()
        digit_toks = extract_digit_tokens(address)
        word_toks = extract_word_tokens(address, min_len=4)

        return (
            self.name_index.query(name_toks) |
            self.digit_index.query(digit_toks) |
            self.word_index.query(word_toks)
        )
