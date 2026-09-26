"""
Data loader module for Business Entity Resolution pipeline.
Handles TSV reading, literal "null" values, string dtypes, and streaming/chunking.
"""

import os
import pandas as pd
from typing import Optional, List, Iterator, Union

def clean_series(series: pd.Series) -> pd.Series:
    """Fills NA, handles case-insensitive literal 'null', and strips whitespace."""
    s = series.fillna("").astype(str).str.strip()
    mask = s.str.lower() == "null"
    if mask.any():
        s = s.mask(mask, "")
    return s

def load_tsv(
    path: str,
    usecols: Optional[List[str]] = None,
    chunksize: Optional[int] = None
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """
    Robust TSV reader adhering to challenge constraints:
    - Tab separator: sep='\\t'
    - String dtypes: dtype=str (prevents ID mangling)
    - Literal 'null' handling: converts 'null' (case-insensitive) to empty string ''
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=usecols,
        keep_default_na=False,
        na_values=[""],
        chunksize=chunksize,
        low_memory=False
    )

    if chunksize is not None:
        def chunk_generator():
            for chunk in reader:
                for col in chunk.columns:
                    chunk[col] = clean_series(chunk[col])
                yield chunk
        return chunk_generator()

    for col in reader.columns:
        reader[col] = clean_series(reader[col])
    return reader

def parse_ground_truth(path: str) -> pd.DataFrame:
    """Loads and parses train_ground_truth.tsv into S1 -> list of matched IDs."""
    df = load_tsv(path, usecols=["source1_entity_id", "matched_entity_ids"])
    def parse_ids(x):
        if not x:
            return []
        return [i.strip() for i in x.split(",") if i.strip()]
    df["matched_ids"] = df["matched_entity_ids"].apply(parse_ids)
    return df
