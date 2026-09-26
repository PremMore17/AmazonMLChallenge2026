"""
EDA / hypothesis-testing script for the Business Entity Resolution challenge.
Run locally from D:\\AmazonMLChallenge2026 with:  python eda_profile.py

Requires only pandas. Reads train + test files, reports:
  - shapes, dtypes, missingness (incl. literal "null" strings)
  - entity_id prefix/length sanity
  - country distribution per file
  - ground truth match-count distribution (singleton %, multi-match %)
  - CROSS-COUNTRY MATCH TEST: does country ever differ between an S1 entity
    and its true matches? (tests whether country is a safe blocking key)
  - script/language mix: fraction of business_name chars outside basic Latin,
    broken down by country
  - name/address length stats
"""
import pandas as pd
import re
import unicodedata
from collections import Counter

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

BASE = "dataset"

def load(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    return df

def report_missing(df, name, cols=("business_name", "business_address", "country")):
    print(f"\n--- {name} ---")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    for c in cols:
        if c not in df.columns:
            continue
        n_null = df[c].isna().sum()
        n_literal_null = (df[c].astype(str).str.strip().str.lower() == "null").sum()
        n_empty = (df[c].astype(str).str.strip() == "").sum()
        print(f"  {c}: NaN={n_null}  literal-'null'={n_literal_null}  empty-string={n_empty}")

def id_prefix_stats(df, expected_prefix, name):
    ids = df["entity_id"].astype(str)
    prefixes = ids.str.extract(r"^([A-Za-z0-9]+-)")[0]
    bad = ids[~ids.str.startswith(expected_prefix)]
    lengths = ids.str.replace(expected_prefix, "", regex=False).str.len()
    print(f"\n--- {name} entity_id check ---")
    print("  unexpected prefixes (should be empty):", bad.tolist()[:5])
    print("  numeric-part length distribution:\n", lengths.value_counts().sort_index())
    print("  duplicate entity_ids:", ids.duplicated().sum())

def country_dist(df, name):
    print(f"\n--- {name} country distribution ---")
    print(df["country"].value_counts(dropna=False))

def latin_fraction(s):
    if not isinstance(s, str) or not s:
        return 1.0
    total = 0
    latin = 0
    for ch in s:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        total += 1
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "LATIN" in name:
            latin += 1
    return latin / total if total else 1.0

def script_mix_report(df, name):
    print(f"\n--- {name} script mix (business_name) by country ---")
    tmp = df.copy()
    tmp["latin_frac"] = tmp["business_name"].apply(latin_fraction)
    tmp["mostly_nonlatin"] = tmp["latin_frac"] < 0.5
    print(tmp.groupby("country")["mostly_nonlatin"].mean())
    print("overall non-Latin-majority rate:", tmp["mostly_nonlatin"].mean())

def length_stats(df, name):
    print(f"\n--- {name} length stats ---")
    for c in ("business_name", "business_address"):
        lens = df[c].fillna("").astype(str).str.len()
        print(f"  {c}: mean={lens.mean():.1f} median={lens.median():.0f} "
              f"p5={lens.quantile(.05):.0f} p95={lens.quantile(.95):.0f} max={lens.max()}")

def main():
    s1 = load(f"{BASE}/train/train_source1.tsv")
    s2 = load(f"{BASE}/train/train_source2.tsv")
    s3 = load(f"{BASE}/train/train_source3.tsv")
    gt = load(f"{BASE}/train/train_ground_truth.tsv")

    ts1 = load(f"{BASE}/test/test_source1.tsv")
    ts2 = load(f"{BASE}/test/test_source2.tsv")
    ts3 = load(f"{BASE}/test/test_source3.tsv")

    for df, nm in [(s1, "train_source1"), (s2, "train_source2"), (s3, "train_source3"),
                   (ts1, "test_source1"), (ts2, "test_source2"), (ts3, "test_source3")]:
        report_missing(df, nm)
        country_dist(df, nm)
        length_stats(df, nm)
        script_mix_report(df, nm)

    id_prefix_stats(s1, "S1-", "train_source1")
    id_prefix_stats(s2, "S2-", "train_source2")
    id_prefix_stats(s3, "S3-", "train_source3")
    id_prefix_stats(ts1, "S1-", "test_source1")
    id_prefix_stats(ts2, "S2-", "test_source2")
    id_prefix_stats(ts3, "S3-", "test_source3")

    # ---- Ground truth analysis ----
    print("\n=== GROUND TRUTH ANALYSIS ===")
    print("total S1 entities in gt:", len(gt))
    print("duplicate source1_entity_id rows:", gt["source1_entity_id"].duplicated().sum())

    def parse_ids(x):
        if pd.isna(x) or str(x).strip() == "":
            return []
        return [i.strip() for i in str(x).split(",") if i.strip()]

    gt["match_list"] = gt["matched_entity_ids"].apply(parse_ids)
    gt["n_matches"] = gt["match_list"].apply(len)
    print("\nmatches-per-S1 distribution:\n", gt["n_matches"].value_counts().sort_index())
    print("singleton rate (0 matches):", (gt["n_matches"] == 0).mean())
    print("multi-match rate (>1 matches):", (gt["n_matches"] > 1).mean())

    gt["n_s2"] = gt["match_list"].apply(lambda lst: sum(1 for i in lst if i.startswith("S2-")))
    gt["n_s3"] = gt["match_list"].apply(lambda lst: sum(1 for i in lst if i.startswith("S3-")))
    print("total S2 matches:", gt["n_s2"].sum(), " total S3 matches:", gt["n_s3"].sum())

    # ---- CROSS-COUNTRY MATCH TEST (key hypothesis) ----
    print("\n=== CROSS-COUNTRY MATCH TEST ===")
    s1_country = dict(zip(s1["entity_id"], s1["country"]))
    s2_country = dict(zip(s2["entity_id"], s2["country"]))
    s3_country = dict(zip(s3["entity_id"], s3["country"]))

    mismatches = 0
    total_pairs = 0
    mismatch_examples = []
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        c1 = s1_country.get(s1_id)
        for mid in row["match_list"]:
            total_pairs += 1
            if mid.startswith("S2-"):
                c2 = s2_country.get(mid)
            elif mid.startswith("S3-"):
                c2 = s3_country.get(mid)
            else:
                c2 = None
            if c1 is not None and c2 is not None and c1 != c2:
                mismatches += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append((s1_id, c1, mid, c2))

    print(f"total ground-truth match pairs checked: {total_pairs}")
    print(f"cross-country mismatches: {mismatches}  "
          f"({100*mismatches/total_pairs:.4f}% of pairs)" if total_pairs else "n/a")
    print("example mismatches (s1_id, s1_country, match_id, match_country):")
    for ex in mismatch_examples:
        print("  ", ex)

if __name__ == "__main__":
    main()