"""
Candidate generation module for Business Entity Resolution pipeline.
Generates candidate pairs per S1 entity using partitioned inverted indices.
Produces candidate pair mapping or candidate_pairs.tsv format.
"""

import os
import sys
import time
from typing import Dict, Set, List, Iterator, Optional
import pandas as pd

try:
    from .blocking import CountryPartitionBlocking
    from .data_loader import load_tsv
except ImportError:
    from blocking import CountryPartitionBlocking
    from data_loader import load_tsv

def build_country_indices(
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    country: str,
    name_cap: int = 2000,
    digit_cap: int = 5000,
    word_cap: int = 2000
) -> CountryPartitionBlocking:
    """Builds and finalizes CountryPartitionBlocking index for a specific country."""
    engine = CountryPartitionBlocking(
        country=country,
        name_cap=name_cap,
        digit_cap=digit_cap,
        word_cap=word_cap
    )
    if not s2_df.empty:
        engine.index_records(s2_df)
    if not s3_df.empty:
        engine.index_records(s3_df)
    engine.finalize()
    return engine

def generate_candidates_for_s1(
    s1_df: pd.DataFrame,
    blocking_engine: CountryPartitionBlocking
) -> Dict[str, Set[str]]:
    """Generates candidate sets for all S1 records in a single country partition."""
    candidates = {}
    for row in s1_df.itertuples(index=False):
        cands = blocking_engine.get_candidates(row.business_name, row.business_address)
        candidates[row.entity_id] = cands
    return candidates

def format_candidate_list(cands: Set[str]) -> str:
    """Formats set of candidate IDs as comma-separated string, sorted for determinism."""
    if not cands:
        return ""
    return ",".join(sorted(cands))
