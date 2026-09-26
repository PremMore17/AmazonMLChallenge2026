"""
Normalization module for Business Entity Resolution.
Implements:
  - clean_text(s)
  - normalize_name(s)
  - normalize_address(s)
  - extract_digit_tokens(address)
  - extract_word_tokens(text, min_len=4)
"""

import os
import re
import unicodedata
import random
from typing import Set, List, Optional
import pandas as pd

# Legal entity tokens to strip from both ends
LEGAL_TOKENS = {
    "llc", "inc", "corp", "corporation", "ltd", "limited", "pvt", "private",
    "plc", "sarl", "sas", "co", "company", "llp", "pc", "lc", "gmbh", "sa",
    "sci", "incorporated", "enterprises", "associates", "holding", "holdings"
}

# Common address abbreviation normalization mappings
ADDRESS_ABBR_MAP = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pkwy": "parkway",
    "hwy": "highway",
    "pl": "place",
    "cir": "circle",
    "ter": "terrace",
    "sq": "square",
    "apt": "apartment",
    "ste": "suite",
    "bldg": "building",
    "fl": "floor",
    "flr": "floor",
    "twp": "township",
    "opp": "opposite",
    "nr": "near",
    "bh": "behind",
    "ext": "extension",
    "extn": "extension",
    "sec": "sector",
    "dist": "district",
    "no": "number",
    "r": "rue"
}

RE_DIGITS = re.compile(r"\d{3,}")
RE_PUNCT = re.compile(r"[^\w\s]")
RE_SPACES = re.compile(r"\s+")
RE_WORD_TOKENS = re.compile(r"[a-zA-Z0-9]+")

_CHAR_CACHE = {}

def latin_fraction(s: Optional[str]) -> float:
    """Calculates fraction of non-whitespace/non-punctuation characters in Latin script."""
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

def is_latin_majority(s: Optional[str], threshold: float = 0.5) -> bool:
    return latin_fraction(s) >= threshold


def strip_punctuation_preserve_marks(s: str) -> str:
    """Replace punctuation/symbol characters with a space, but keep
    Letters (L*), Marks (M*), Numbers (N*), and whitespace untouched."""
    out = []
    for ch in s:
        if ch.isspace():
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "M", "N"):
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def clean_text(s: Optional[str], debug: bool = False) -> str:
    """
    Handle NaN and literal case-insensitive 'null' as empty string;
    Remove standalone 'null' tokens anywhere in the string;
    Collapse dotted acronyms (L.L.C. -> LLC, U.S.A. -> USA);
    Diacritic folding ONLY on Latin-majority text (NFKD + filter Mn combining marks: Mált -> malt, Láne -> lane);
    Non-Latin text preserves all characters and combining marks (e.g. Indic matras);
    Unicode NFKC normalize; lowercase; collapse whitespace/punctuation.
    """
    if debug:
        print(f"  [clean_text:input] {s!r}")
    if s is None or pd.isna(s):
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s or s.lower() == "null":
        return ""

    # Remove standalone literal 'null' tokens anywhere in the string
    s = re.sub(r"(?i)\bnull\b", " ", s)
    if debug:
        print(f"  [clean_text:after_null_strip] {s!r}")

    # Dotted acronym collapse before punctuation removal: e.g. L.L.C. -> LLC, P.V.T. -> PVT
    s = re.sub(r"\b(?:[a-zA-Z]\.)+[a-zA-Z]\.?", lambda m: m.group(0).replace(".", ""), s)
    s = re.sub(r"\b(?:[a-zA-Z]\.){2,}", lambda m: m.group(0).replace(".", ""), s)
    if debug:
        print(f"  [clean_text:after_acronym_collapse] {s!r}")

    # Diacritic folding: ONLY apply NFKD-decompose-and-strip-Mn if Latin-majority
    # Preserves essential Indic vowel signs (matras/viramas) in Devanagari, Kannada, Tamil, etc.
    is_lat = is_latin_majority(s, threshold=0.5)
    if debug:
        print(f"  [clean_text:is_latin_majority] {is_lat} (fraction={latin_fraction(s):.2f})")
    if is_lat:
        s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if unicodedata.category(ch) != "Mn")
        s = unicodedata.normalize("NFKC", s).lower()
    else:
        s = unicodedata.normalize("NFKC", s).lower()
    if debug:
        print(f"  [clean_text:after_unicode_branch] {s!r}")

    # Normalize & to and
    s = re.sub(r"\s*&\s*", " and ", s)
    if debug:
        print(f"  [clean_text:after_ampersand] {s!r}")

    # Replace punctuation with whitespace, preserving Unicode Marks (matras/combining marks)
    s = strip_punctuation_preserve_marks(s)
    if debug:
        print(f"  [clean_text:after_strip_punct] {s!r}")

    # Collapse multiple whitespaces
    s = RE_SPACES.sub(" ", s).strip()
    if debug:
        print(f"  [clean_text:after_space_collapse] {s!r}")
    return s


def normalize_name(s: Optional[str], debug: bool = False) -> str:
    """
    Apply clean_text, then strip legal-entity token set from BOTH ends
    (llc, inc, corp, corporation, ltd, limited, pvt, private, plc, sarl, sas, co, company).
    Preserves raw text if stripping empties the string.
    """
    if debug:
        print(f"[normalize_name:start] Input: {s!r}")
    cleaned = clean_text(s, debug=debug)
    if debug:
        print(f"[normalize_name:after_clean_text] {cleaned!r}")
    if not cleaned:
        return ""
    
    tokens = cleaned.split()
    if debug:
        print(f"[normalize_name:tokenization] {tokens!r}")
    
    # Strip legal entity tokens from the front
    changed = True
    while changed and tokens:
        changed = False
        if tokens[0] in LEGAL_TOKENS:
            if debug:
                print(f"[normalize_name:strip_front_legal] Popped {tokens[0]!r}")
            tokens.pop(0)
            changed = True

    # Strip legal entity tokens from the back
    changed = True
    while changed and tokens:
        changed = False
        if tokens[-1] in LEGAL_TOKENS:
            if debug:
                print(f"[normalize_name:strip_back_legal] Popped {tokens[-1]!r}")
            tokens.pop(-1)
            changed = True
            
    if not tokens:
        if debug:
            print(f"[normalize_name:final_rejoin] Tokens empty, returning cleaned: {cleaned!r}")
        return cleaned
    result = " ".join(tokens)
    if debug:
        print(f"[normalize_name:final_rejoin] Result: {result!r}")
    return result

def normalize_address(s: Optional[str], debug: bool = False) -> str:
    """
    Apply clean_text, then normalize common address abbreviations (rd/road, st/street, etc.).
    """
    if debug:
        print(f"[normalize_address:start] Input: {s!r}")
    cleaned = clean_text(s, debug=debug)
    if debug:
        print(f"[normalize_address:after_clean_text] {cleaned!r}")
    if not cleaned:
        return ""
    
    tokens = cleaned.split()
    if debug:
        print(f"[normalize_address:tokenization] {tokens!r}")
    normalized_tokens = [ADDRESS_ABBR_MAP.get(tok, tok) for tok in tokens]
    if debug:
        print(f"[normalize_address:abbr_expansion] {normalized_tokens!r}")
    result = " ".join(normalized_tokens)
    if debug:
        print(f"[normalize_address:final_rejoin] Result: {result!r}")
    return result

def extract_digit_tokens(address: Optional[str]) -> Set[str]:
    """
    Extract all digit sequences of length >= 3 using regex \\d{3,}.
    Returns a set of matched digit strings.
    """
    if address is None or pd.isna(address) or not isinstance(address, str):
        return set()
    return set(RE_DIGITS.findall(address))

def extract_word_tokens(text: Optional[str], min_len: int = 4) -> Set[str]:
    """
    Tokenize on whitespace/punctuation, keep only tokens of length >= min_len,
    lowercase, and ASCII/Latin only.
    """
    if text is None or pd.isna(text) or not isinstance(text, str):
        return set()
    
    # Extract alphanumeric word tokens
    words = RE_WORD_TOKENS.findall(text)
    valid_tokens = set()
    for w in words:
        if len(w) >= min_len and w.isascii() and any(c.isalpha() for c in w):
            valid_tokens.add(w.lower())
    return valid_tokens

def run_sample_check():
    """Sample 2000 rows per train source, print 15 random before/after examples for sanity checking."""
    print("=" * 80)
    print("NORMALIZATION SANITY CHECK — 2,000 ROWS PER SOURCE")
    print("=" * 80)

    # Unit verification of fixes
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
        ("123 Main St., Suite #4 (rear)", normalize_address, "123 main street suite 4 rear"),
        (float("nan"), clean_text, ""),
        ("null", clean_text, ""),
        (None, clean_text, "")
    ]
    for inp, fn, expected in unit_tests:
        out = fn(inp)
        status = "PASS" if out == expected else f"FAIL (got {out!r})"
        print(f"  [{status}] {fn.__name__}({inp!r}) -> {out!r}")

    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "dataset", "train")
    if not os.path.exists(dataset_dir):
        dataset_dir = os.path.join("dataset", "train")

    random.seed(42)

    for src in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(dataset_dir, src)
        if not os.path.isfile(path):
            print(f"Skipping {src}: not found.")
            continue
        print(f"\n--- Reading sample from {src} ---")
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
        sample_names = random.sample(name_examples, min(15, len(name_examples))) if name_examples else []
        for orig, norm in sample_names:
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
        sample_addrs = random.sample(addr_examples, min(15, len(addr_examples))) if addr_examples else []
        for orig, norm in sample_addrs:
            print(f"  BEFORE: {orig!r}")
            print(f"  AFTER : {norm!r}")

        # 5 examples for extract_digit_tokens and extract_word_tokens
        print(f"\n[Token Extraction] 5 Examples from {src}:")
        count = 0
        for addr in sample["business_address"]:
            if pd.isna(addr) or not addr or str(addr).strip().lower() == "null":
                continue
            digits = extract_digit_tokens(addr)
            words = extract_word_tokens(addr, min_len=4)
            print(f"  ADDR: {addr!r}")
            print(f"  DIGITS (\\d{{3,}}): {sorted(list(digits))}")
            print(f"  WORD TOKENS (len>=4, ASCII): {sorted(list(words))}")
            count += 1
            if count >= 5:
                break

if __name__ == "__main__":
    run_sample_check()
