# utils/data_loader.py — Parquet-first loader with CSV fallback + metadata + PS-wide in-memory preload
from __future__ import annotations

import os
import time
import shutil
from typing import Iterable, Optional, Sequence

import pandas as pd
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

# Google Drive CSV (set real ID in Streamlit secrets as RESULTS2024_FILE_ID)
GDRIVE_FILE_ID_FALLBACK = ""  # optional placeholder; prefer st.secrets["RESULTS2024_FILE_ID"]
LOCAL_GZ_PATH = os.environ.get("PSES_RESULTS_GZ", "/tmp/Results2024.csv.gz")

# Parquet dataset location (directory). Prefer a persistent folder.
PARQUET_ROOTDIR = os.environ.get("PSES_PARQUET_DIR", "data/parquet/PSES_Results2024_v2")
PARQUET_FLAG = os.path.join(PARQUET_ROOTDIR, "_BUILD_OK_v2")

# Output schema (normalized)
OUT_COLS = [
    "year", "question_code", "group_value", "n",
    "positive_pct", "neutral_pct", "negative_pct", "agree_pct",
    "answer1", "answer2", "answer3", "answer4", "answer5", "answer6", "answer7",
]

DTYPES = {
    "year": "Int16",
    "question_code": "string",
    "group_value": "string",
    "n": "Int32",
    "positive_pct": "Float32",
    "neutral_pct": "Float32",
    "negative_pct": "Float32",
    "agree_pct": "Float32",
    "answer1": "Float32", "answer2": "Float32", "answer3": "Float32",
    "answer4": "Float32", "answer5": "Float32", "answer6": "Float32", "answer7": "Float32",
}

# Minimal column set to read from CSV (include LEVEL1ID for PS-wide filter)
CSV_USECOLS = [
    "LEVEL1ID",  # may be missing in some exports; handled defensively
    "SURVEYR", "QUESTION", "DEMCODE",
    "ANSCOUNT", "POSITIVE", "NEUTRAL", "NEGATIVE", "AGREE",
    "answer1", "answer2", "answer3", "answer4", "answer5", "answer6", "answer7",
]

# =============================================================================
# Internal diagnostics (visible in Status & Menu 1 footer)
# =============================================================================
_LAST_DIAG: dict = {}
_LAST_ENGINE: str = "unknown"
_AGREE_SRC: Optional[str] = None  # detected CSV header used for agree_pct (e.g., "AGREE" or "AGREE_PCT")

def _set_diag(**kwargs):
    _LAST_DIAG.clear()
    _LAST_DIAG.update(kwargs)

def get_last_query_diag() -> dict:
    """Diagnostics for most recent load_results2024_filtered call."""
    return dict(_LAST_DIAG)

# =============================================================================
# Capability checks
# =============================================================================
def _duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401
        return True
    except Exception:
        return False

def _pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.dataset as ds  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
        return True
    except Exception:
        return False

# =============================================================================
# CSV presence (cached)
# =============================================================================
@st.cache_resource(show_spinner="📥 Downloading Results2024.csv.gz…")
def ensure_results2024_local(file_id: Optional[str] = None) -> str:
    import gdown
    fid = file_id or st.secrets.get("RESULTS2024_FILE_ID", GDRIVE_FILE_ID_FALLBACK)
    if not fid:
        raise RuntimeError("RESULTS2024_FILE_ID missing in .streamlit/secrets.toml")

    if os.path.exists(LOCAL_GZ_PATH) and os.path.getsize(LOCAL_GZ_PATH) > 0:
        return LOCAL_GZ_PATH

    os.makedirs(os.path.dirname(LOCAL_GZ_PATH), exist_ok=True)
    url = f"https://drive.google.com/uc?id={fid}"
    gdown.download(url, LOCAL_GZ_PATH, quiet=False)
    if not os.path.exists(LOCAL_GZ_PATH) or os.path.getsize(LOCAL_GZ_PATH) == 0:
        raise RuntimeError("Download failed or produced an empty file.")
    return LOCAL_GZ_PATH

# =============================================================================
# Agree header detection
# =============================================================================
def _detect_agree_header(csv_path: str) -> Optional[str]:
    """
    Detect the Agree-like column name in the CSV header.
    Returns the first match among common variants; else None.
    """
    try:
        cols = pd.read_csv(csv_path, compression="gzip", nrows=0).columns.tolist()
    except Exception:
        return None
    # Build a case-insensitive lookup
    lower_map = {c.lower(): c for c in cols}

    # Preferred matches (first wins)
    candidates = [
        "agree", "agree_pct", "agree pct", "agreepercent", "pct_agree"
    ]
    for key in candidates:
        if key in lower_map:
            return lower_map[key]
    return None

def _with_agree_in_usecols(base: Sequence[str], csv_path: str) -> list[str]:
    """
    Return a copy of base usecols with the detected Agree header added if present.
    """
    agree_col = _detect_agree_header(csv_path)
    # update module-level diagnostic source
    global _AGREE_SRC
    _AGREE_SRC = agree_col
    s = {c for c in base}
    if agree_col:
        s.add(agree_col)
    return list(s)

def _csv_has_level1id(csv_path: str) -> bool:
    """Peek header to see if LEVEL1ID exists (robust for DuckDB path)."""
    try:
        cols = pd.read_csv(csv_path, compression="gzip", nrows=0).columns
        return "LEVEL1ID" in cols
    except Exception:
        return False

# =============================================================================
# Canonicalization helpers
# =============================================================================
def _canon_demcode_series(s: pd.Series) -> pd.Series:
    """
    Canonicalize DEMCODE/group_value as a string:
      - trim whitespace
      - remove trailing .0 / .000 (e.g., 8474.0 -> 8474)
      - remove trailing zeros from fractional part (8474.50 -> 8474.5)
      - remove trailing dot (123. -> 123)
      - blanks/NaN -> "All"
    """
    s = s.astype("string")
    s = s.str.strip()
    # drop .0... entirely
    s = s.str.replace(r"\.0+$", "", regex=True)
    # drop trailing zeros in fractional part but keep at least one significant (e.g., .50 -> .5)
    s = s.str.replace(r"(\.\d*?[1-9])0+$", r"\1", regex=True)
    # drop dangling dot
    s = s.str.replace(r"\.$", "", regex=True)
    # normalize blanks to "All"
    s = s.mask(s.isna() | (s == ""), "All")
    return s

def _canon_demcode_value(v: Optional[str]) -> Optional[str]:
    """Canonicalize a single DEMCODE value (used on the filter target)."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "all":
        return None  # None means overall in our filter API
    # same transforms as the series helper
    import re
    s = re.sub(r"\.0+$", "", s)
    s = re.sub(r"(\.\d*?[1-9])0+$", r"\1", s)
    s = re.sub(r"\.$", "", s)
    return s

# =============================================================================
# Type normalization
# =============================================================================
def _normalize_df_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final, canonical normalization applied to all code paths:
      - year           → Int16
      - question_code  → string, TRIM, UPPER
      - group_value    → canonicalized string (8474.0 -> 8474), blanks/NaN → "All"
      - metrics        → typed
    """
    if df is None or df.empty:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame(columns=OUT_COLS)

    df = df.reindex(columns=OUT_COLS)

    # year/numeric columns
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(DTYPES["year"])
    df["n"]    = pd.to_numeric(df["n"], errors="coerce").astype(DTYPES["n"])
    for c in ["positive_pct","neutral_pct","negative_pct","agree_pct",
              "answer1","answer2","answer3","answer4","answer5","answer6","answer7"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(DTYPES[c])

    # question_code: UPPER(TRIM)
    q = df["question_code"].astype("string")
    df["question_code"] = q.str.strip().str.upper().astype(DTYPES["question_code"])

    # group_value: canonicalize (handles 8474.0 -> 8474 and blanks -> "All")
    df["group_value"] = _canon_demcode_series(df["group_value"]).astype(DTYPES["group_value"])

    return df

# =============================================================================
# Build Parquet one-time (filter to PS-wide iff LEVEL1ID exists)
# =============================================================================
def _build_parquet_with_duckdb(csv_path: str) -> None:
    import duckdb
    os.makedirs(PARQUET_ROOTDIR, exist_ok=True)
    con = duckdb.connect()

    has_lvl1 = _csv_has_level1id(csv_path)
    where_clause = "WHERE CAST(LEVEL1ID AS INT) = 0" if has_lvl1 else ""

    # Detect agree header once
    agree_col = _detect_agree_header(csv_path)
    # Update diagnostic
    global _AGREE_SRC
    _AGREE_SRC = agree_col

    # Agree projection: either cast detected column or NULL if absent
    if agree_col:
        agree_select = f'CAST("{agree_col}" AS DOUBLE)                                     AS agree_pct,'
    else:
        agree_select = 'CAST(NULL AS DOUBLE)                                               AS agree_pct,'

    # Note: DEMCODE canonicalization is applied via regex_replace in SQL.
    con.execute(f"""
        CREATE OR REPLACE TABLE pses AS
        SELECT
          CAST(SURVEYR AS INT)                                         AS year,
          UPPER(TRIM(CAST(QUESTION AS VARCHAR)))                       AS question_code,
          COALESCE(
            NULLIF(
              REGEXP_REPLACE(
                REGEXP_REPLACE(TRIM(CAST(DEMCODE AS VARCHAR)), '(\\.[0-9]*?[1-9])0+$', '\\1'),
                '\\.0+$', ''
              ),
              ''
            ),
            'All'
          )                                                             AS group_value,
          CAST(ANSCOUNT AS INT)                                        AS n,
          CAST(POSITIVE AS DOUBLE)                                     AS positive_pct,
          CAST(NEUTRAL  AS DOUBLE)                                     AS neutral_pct,
          CAST(NEGATIVE AS DOUBLE)                                     AS negative_pct,
          {agree_select}
          CAST(answer1  AS DOUBLE) AS answer1,
          CAST(answer2  AS DOUBLE) AS answer2,
          CAST(answer3  AS DOUBLE) AS answer3,
          CAST(answer4  AS DOUBLE) AS answer4,
          CAST(answer5  AS DOUBLE) AS answer5,
          CAST(answer6  AS DOUBLE) AS answer6,
          CAST(answer7  AS DOUBLE) AS answer7
        FROM read_csv_auto(?, header=true)
        {where_clause}
    """, [csv_path])

    con.execute(f"""
        COPY pses TO '{PARQUET_ROOTDIR}'
        (FORMAT PARQUET, COMPRESSION 'ZSTD', ROW_GROUP_SIZE 1000000,
         PARTITION_BY (year, question_code));
    """)

def _build_parquet_with_pandas(csv_path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Detect agree header and build usecols dynamically so we don't drop it
    agree_col = _detect_agree_header(csv_path)
    global _AGREE_SRC
    _AGREE_SRC = agree_col

    base_cols = [c for c in CSV_USECOLS if c != "LEVEL1ID"]
    usecols = _with_agree_in_usecols(base_cols, csv_path)

    df = pd.read_csv(csv_path, compression="gzip", usecols=usecols, low_memory=False)

    # Bring LEVEL1ID if present (separate small pass)
    try:
        cols = pd.read_csv(csv_path, compression="gzip", nrows=0).columns
        if "LEVEL1ID" in cols:
            df2 = pd.read_csv(csv_path, compression="gzip", usecols=["LEVEL1ID"], low_memory=True)
            df["LEVEL1ID"] = df2["LEVEL1ID"]
    except Exception:
        pass

    # Filter to PS-wide if LEVEL1ID present
    if "LEVEL1ID" in df.columns:
        df = df[pd.to_numeric(df["LEVEL1ID"], errors="coerce").fillna(0).astype(int).eq(0)]

    # Normalize to OUT_COLS (apply TRIM/UPPER + DEMCODE canonicalization)
    out = pd.DataFrame({
        "year":          pd.to_numeric(df["SURVEYR"], errors="coerce").astype("Int64"),
        "question_code": df["QUESTION"].astype("string").str.strip().str.upper(),
        "group_value":   _canon_demcode_series(df["DEMCODE"]),
        "n":             pd.to_numeric(df["ANSCOUNT"], errors="coerce").astype("Int64"),
        "positive_pct":  pd.to_numeric(df["POSITIVE"], errors="coerce"),
        "neutral_pct":   pd.to_numeric(df["NEUTRAL"],  errors="coerce"),
        "negative_pct":  pd.to_numeric(df["NEGATIVE"], errors="coerce"),
        "agree_pct":     pd.to_numeric(df.get(agree_col) if agree_col else None, errors="coerce"),
        "answer1": pd.to_numeric(df.get("answer1"), errors="coerce"),
        "answer2": pd.to_numeric(df.get("answer2"), errors="coerce"),
        "answer3": pd.to_numeric(df.get("answer3"), errors="coerce"),
        "answer4": pd.to_numeric(df.get("answer4"), errors="coerce"),
        "answer5": pd.to_numeric(df.get("answer5"), errors="coerce"),
        "answer6": pd.to_numeric(df.get("answer6"), errors="coerce"),
        "answer7": pd.to_numeric(df.get("answer7"), errors="coerce"),
    })

    os.makedirs(PARQUET_ROOTDIR, exist_ok=True)
    table = pa.Table.from_pandas(out[OUT_COLS], preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=PARQUET_ROOTDIR,
        partition_cols=["year", "question_code"],
        compression="zstd",
    )

def _parquet_rowcount(root: str) -> int:
    try:
        import pyarrow.dataset as ds
        dataset = ds.dataset(root, format="parquet")
        return int(dataset.count_rows())
    except Exception:
        return 0

@st.cache_resource(show_spinner="🗂️ Preparing Parquet dataset (one-time)…")
def ensure_parquet_dataset() -> str:
    """
    Ensures a partitioned Parquet dataset (PS-wide if LEVEL1ID present) exists and is non-empty.
    If an existing dataset is empty, it gets rebuilt automatically.
    """
    if not _pyarrow_available():
        raise RuntimeError("pyarrow is required for Parquet fast path.")
    csv_path = ensure_results2024_local()

    # Touch detection early for diagnostics
    try:
        global _AGREE_SRC
        _AGREE_SRC = _detect_agree_header(csv_path)
    except Exception:
        pass

    # If dataset looks ready, validate it's not empty
    if os.path.isdir(PARQUET_ROOTDIR) and os.path.exists(PARQUET_FLAG):
        if _parquet_rowcount(PARQUET_ROOTDIR) > 0:
            return PARQUET_ROOTDIR
        # Rebuild if empty
        try:
            shutil.rmtree(PARQUET_ROOTDIR, ignore_errors=True)
        except Exception:
            pass

    os.makedirs(PARQUET_ROOTDIR, exist_ok=True)

    if _duckdb_available():
        _build_parquet_with_duckdb(csv_path)
    else:
        _build_parquet_with_pandas(csv_path)

    # Validate row count; if still zero, raise so caller can fall back
    rc = _parquet_rowcount(PARQUET_ROOTDIR)
    if rc <= 0:
        raise RuntimeError("Parquet dataset built but contains 0 rows.")

    with open(PARQUET_FLAG, "w") as f:
        f.write("ok")
    return PARQUET_ROOTDIR

# =============================================================================
# Metadata loaders (cached)
# =============================================================================
def _path_candidate(*names: str) -> str | None:
    for n in names:
        if n and os.path.exists(n):
            return n
    return None

@st.cache_resource(show_spinner=False)
def load_questions_metadata() -> pd.DataFrame:
    """
    Reads metadata/Survey Questions.xlsx
    - 'question' column contains the question number/code (e.g., Q16)
    - 'English' column contains the question text
    """
    path = _path_candidate("metadata/Survey Questions.xlsx", "/mnt/data/Survey Questions.xlsx")
    if not path:
        return pd.DataFrame(columns=["code", "text", "display"])
    qdf = pd.read_excel(path)
    cols = {c.lower(): c for c in qdf.columns}
    code_col = cols.get("question")
    text_col = cols.get("english")
    if not code_col or not text_col:
        return pd.DataFrame(columns=["code", "text", "display"])
    out = pd.DataFrame({
        "code": qdf[code_col].astype(str).str.strip(),
        "text": qdf[text_col].astype(str),
    })
    out["qnum"] = out["code"].str.extract(r"Q?(\d+)", expand=False)
    with pd.option_context("mode.chained_assignment", None):
        out["qnum"] = pd.to_numeric(out["qnum"], errors="coerce")
    out = out.sort_values(["qnum", "code"], na_position="last")
    out["display"] = out["code"] + " – " + out["text"].astype(str)
    return out[["code", "text", "display"]]

@st.cache_resource(show_spinner=False)
def load_scales_metadata() -> pd.DataFrame:
    path = _path_candidate("metadata/Survey Scales.xlsx", "/mnt/data/Survey Scales.xlsx")
    if not path:
        return pd.DataFrame()
    sdf = pd.read_excel(path)
    sdf.columns = sdf.columns.str.strip().str.lower()
    def _normalize_qcode(s: str) -> str:
        s = "" if s is None else str(s)
        s = s.upper()
        return "".join(ch for ch in s if ch.isalnum())
    code_col = None
    for c in ("code", "question"):
        if c in sdf.columns:
            code_col = c
            break
    if code_col:
        sdf["__code_norm__"] = sdf[code_col].astype(str).map(_normalize_qcode)
    return sdf

@st.cache_resource(show_spinner=False)
def load_demographics_metadata() -> pd.DataFrame:
    path = _path_candidate("metadata/Demographics.xlsx", "/mnt/data/Demographics.xlsx")
    if not path:
        return pd.DataFrame()
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df

# =============================================================================
# In-memory PS-wide preload (LEVEL1ID==0; all DEMCODEs preserved)
# =============================================================================
@st.cache_resource(show_spinner="🧠 Loading PS-wide data into memory…")
def preload_pswide_dataframe() -> pd.DataFrame:
    """
    Loads the PS-wide (LEVEL1ID==0) slice into memory with normalized OUT_COLS.
    Prefer Parquet (already filtered at build); fall back to CSV streaming if empty.
    """
    # Prefer Parquet; ensure valid & non-empty
    try:
        root = ensure_parquet_dataset()
        if _pyarrow_available():
            import pyarrow.dataset as ds
            dataset = ds.dataset(root, format="parquet")
            table = dataset.to_table(columns=OUT_COLS)
            df = table.to_pandas(types_mapper=pd.ArrowDtype)
            if df is not None and not df.empty:
                # final canonical normalization (idempotent)
                return _normalize_df_types(df)
    except Exception:
        # Fall through to CSV preload
        pass

    # CSV streaming preload (PS-wide filter if LEVEL1ID present)
    try:
        path = ensure_results2024_local()
        base_usecols = _with_agree_in_usecols(CSV_USECOLS, path)
        frames: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, compression="gzip", usecols=base_usecols, chunksize=1_500_000, low_memory=True):
            if "LEVEL1ID" in chunk.columns:
                mask_lvl = pd.to_numeric(chunk["LEVEL1ID"], errors="coerce").fillna(0).astype(int).eq(0)
                chunk = chunk.loc[mask_lvl, :]
            sel = chunk
            agree_col = _AGREE_SRC  # populated by _with_agree_in_usecols
            out = pd.DataFrame({
                "year":          pd.to_numeric(sel["SURVEYR"], errors="coerce"),
                "question_code": sel["QUESTION"].astype("string").str.strip().str.upper(),
                "group_value":   _canon_demcode_series(sel["DEMCODE"]),
                "n":             pd.to_numeric(sel["ANSCOUNT"], errors="coerce"),
                "positive_pct":  pd.to_numeric(sel["POSITIVE"], errors="coerce"),
                "neutral_pct":   pd.to_numeric(sel["NEUTRAL"],  errors="coerce"),
                "negative_pct":  pd.to_numeric(sel["NEGATIVE"], errors="coerce"),
                "agree_pct":     pd.to_numeric(sel.get(agree_col) if agree_col else None, errors="coerce"),
                "answer1": pd.to_numeric(sel.get("answer1"), errors="coerce"),
                "answer2": pd.to_numeric(sel.get("answer2"), errors="coerce"),
                "answer3": pd.to_numeric(sel.get("answer3"), errors="coerce"),
                "answer4": pd.to_numeric(sel.get("answer4"), errors="coerce"),
                "answer5": pd.to_numeric(sel.get("answer5"), errors="coerce"),
                "answer6": pd.to_numeric(sel.get("answer6"), errors="coerce"),
                "answer7": pd.to_numeric(sel.get("answer7"), errors="coerce"),
            })
            frames.append(out)
        df = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUT_COLS))
        return _normalize_df_types(df)
    except Exception:
        return pd.DataFrame(columns=OUT_COLS)

# Track in-memory status
_INMEM_STATE: dict = {"mode": "none", "rows": 0}
_META_CACHE: dict = {"counts": {}, "questions": None, "scales": None, "demographics": None}

# =============================================================================
# Parquet query
# =============================================================================
def _parquet_query(question_code: str, years: Iterable[int | str], group_value: Optional[str]) -> pd.DataFrame:
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    root = ensure_parquet_dataset()
    dataset = ds.dataset(root, format="parquet")

    # Compare using normalized value (UPPER/TRIM applied at build)
    q = str(question_code).strip().upper()
    years_int = [int(y) for y in years]
    gv_norm = _canon_demcode_value(group_value)  # canonicalize target

    filt = (pc.field("question_code") == q) & (pc.field("year").isin(years_int))
    if gv_norm is None:
        filt = filt & (pc.field("group_value") == "All")
    else:
        filt = filt & (pc.field("group_value") == gv_norm)

    tbl = dataset.to_table(columns=OUT_COLS, filter=filt)
    df = tbl.to_pandas(types_mapper=pd.ArrowDtype)
    return _normalize_df_types(df)

# =============================================================================
# CSV fallback (PS-wide filter if LEVEL1ID present)
# =============================================================================
def _csv_stream_filter(
    question_code: str,
    years: Iterable[int | str],
    group_value: Optional[str],
    chunksize: int = 1_500_000,
) -> pd.DataFrame:
    path = ensure_results2024_local()
    years_int = [int(y) for y in years]
    q_norm = str(question_code).strip().str.upper() if hasattr(str(question_code), "strip") else str(question_code).upper()
    q_norm = str(question_code).strip().upper()
    gv_target = _canon_demcode_value(group_value)  # canonical form or None (overall)

    frames: list[pd.DataFrame] = []
    usecols = _with_agree_in_usecols(CSV_USECOLS, path)
    for chunk in pd.read_csv(path, compression="gzip", usecols=usecols, chunksize=chunksize, low_memory=True):
        if "LEVEL1ID" in chunk.columns:
            mask_lvl = pd.to_numeric(chunk["LEVEL1ID"], errors="coerce").fillna(0).astype(int).eq(0)
        else:
            mask_lvl = pd.Series(True, index=chunk.index)

        # Normalize QUESTION/YEAR for equality
        q_ser = chunk["QUESTION"].astype(str).str.strip().str.upper()
        y_ser = pd.to_numeric(chunk["SURVEYR"], errors="coerce")

        mask = (q_ser == q_norm) & (y_ser.isin(years_int)) & mask_lvl

        # Canonicalize DEMCODE column before comparing
        gv_ser = _canon_demcode_series(chunk["DEMCODE"])

        if gv_target is None:
            mask &= (gv_ser == "All")
        else:
            mask &= (gv_ser == gv_target)

        if mask.any():
            sel = chunk.loc[mask, :]
            agree_col = _AGREE_SRC
            out = pd.DataFrame({
                "year":          pd.to_numeric(sel["SURVEYR"], errors="coerce"),
                "question_code": sel["QUESTION"].astype("string").str.strip().str.upper(),
                "group_value":   _canon_demcode_series(sel["DEMCODE"]),
                "n":             pd.to_numeric(sel["ANSCOUNT"], errors="coerce"),
                "positive_pct":  pd.to_numeric(sel["POSITIVE"], errors="coerce"),
                "neutral_pct":   pd.to_numeric(sel["NEUTRAL"],  errors="coerce"),
                "negative_pct":  pd.to_numeric(sel["NEGATIVE"], errors="coerce"),
                "agree_pct":     pd.to_numeric(sel.get(agree_col) if agree_col else None, errors="coerce"),
                "answer1": pd.to_numeric(sel.get("answer1"), errors="coerce"),
                "answer2": pd.to_numeric(sel.get("answer2"), errors="coerce"),
                "answer3": pd.to_numeric(sel.get("answer3"), errors="coerce"),
                "answer4": pd.to_numeric(sel.get("answer4"), errors="coerce"),
                "answer5": pd.to_numeric(sel.get("answer5"), errors="coerce"),
                "answer6": pd.to_numeric(sel.get("answer6"), errors="coerce"),
                "answer7": pd.to_numeric(sel.get("answer7"), errors="coerce"),
            })
            frames.append(out)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUT_COLS)
    return _normalize_df_types(df)

# =============================================================================
# Public API — prefers in-memory PS-wide preload; falls back to Parquet/CSV
# =============================================================================
@st.cache_data(show_spinner="🔎 Filtering results…")
def load_results2024_filtered(
    question_code: str,
    years: Iterable[int | str],
    group_value: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns a filtered slice at (question_code, years, group_value) grain.
    Prefers in-memory PS-wide preload (LEVEL1ID==0), then Parquet pushdown, then CSV streaming.
    Records detailed diagnostics for the UI.
    """
    global _LAST_ENGINE
    parquet_error = None
    t0 = time.perf_counter()

    # Try in-memory PS-wide DataFrame first (fast & no I/O)
    try:
        df_all = preload_pswide_dataframe()
        if isinstance(df_all, pd.DataFrame) and not df_all.empty:
            q = str(question_code).strip().upper()
            years_int = [int(y) for y in years]
            gv_norm = _canon_demcode_value(group_value)  # None for overall

            # Build masks using normalized comparisons (df_all already normalized)
            mask = (df_all["question_code"] == q) & (df_all["year"].astype(int).isin(years_int))
            if gv_norm is None:
                mask &= (df_all["group_value"] == "All")
            else:
                mask &= (df_all["group_value"] == gv_norm)

            df = df_all.loc[mask, :]
            _LAST_ENGINE = "inmem"
            rows = int(df.shape[0])
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            _set_diag(
                engine=_LAST_ENGINE,
                elapsed_ms=elapsed_ms,
                rows=rows,
                question_code=str(question_code),
                years=",".join(str(y) for y in years),
                group_value=("All" if gv_norm is None else gv_norm),
                parquet_dir=PARQUET_ROOTDIR,
                csv_path=LOCAL_GZ_PATH,
                parquet_error=None,
                agree_src=_AGREE_SRC,
            )
            return df
    except Exception:
        pass

    # Try Parquet pushdown next
    if _pyarrow_available():
        try:
            df = _parquet_query(question_code, years, group_value)
            _LAST_ENGINE = "parquet"
            rows = int(df.shape[0])
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            _set_diag(
                engine=_LAST_ENGINE,
                elapsed_ms=elapsed_ms,
                rows=rows,
                question_code=str(question_code),
                years=",".join(str(y) for y in years),
                group_value=("All" if _canon_demcode_value(group_value) is None else _canon_demcode_value(group_value)),
                parquet_dir=PARQUET_ROOTDIR,
                csv_path=LOCAL_GZ_PATH,
                parquet_error=None,
                agree_src=_AGREE_SRC,
            )
            return df
        except Exception as e:
            parquet_error = str(e)

    # CSV fallback
    df = _csv_stream_filter(question_code, years, group_value)
    _LAST_ENGINE = "csv"
    rows = int(df.shape[0])
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    _set_diag(
        engine=_LAST_ENGINE,
        elapsed_ms=elapsed_ms,
        rows=rows,
        question_code=str(question_code),
        years=",".join(str(y) for y in years),
        group_value=("All" if _canon_demcode_value(group_value) is None else _canon_demcode_value(group_value)),
        parquet_dir=PARQUET_ROOTDIR,
        csv_path=LOCAL_GZ_PATH,
        parquet_error=parquet_error,
        agree_src=_AGREE_SRC,
    )
    return df

# =============================================================================
# Helpers for UI (Diagnostics / prewarm)
# =============================================================================
def _compute_pswide_rowcount_parquet() -> int:
    try:
        return _parquet_rowcount(PARQUET_ROOTDIR)
    except Exception:
        return 0

def _compute_pswide_rowcount_csv() -> int:
    try:
        path = ensure_results2024_local()
        rows = 0
        for chunk in pd.read_csv(
            path, compression="gzip", usecols=_with_agree_in_usecols(CSV_USECOLS, path), chunksize=2_000_000, low_memory=True
        ):
            if "LEVEL1ID" in chunk.columns:
                mask = pd.to_numeric(chunk["LEVEL1ID"], errors="coerce").fillna(0).astype(int).eq(0)
            else:
                gv = _canon_demcode_series(chunk["DEMCODE"])
                mask = (gv == "All")
            rows += int(mask.sum())
        return rows
    except Exception:
        return 0

@st.cache_resource(show_spinner="⚡ Preloading metadata & PS-wide index…")
def prewarm_all() -> dict:
    """
    Build/ensure backend (Parquet or CSV), load metadata into cache,
    load the PS-wide DataFrame into memory, and return a summary for UI.
    """
    # Ensure raw results are present
    ensure_results2024_local()

    # Prefer Parquet if available (and build if needed)
    engine = "csv"
    try:
        ensure_parquet_dataset()
        engine = "parquet"
    except Exception:
        engine = "csv"

    # Metadata (cached)
    qdf = load_questions_metadata()
    sdf = load_scales_metadata()
    ddf = load_demographics_metadata()
    _META_CACHE["questions"] = qdf
    _META_CACHE["scales"] = sdf
    _META_CACHE["demographics"] = ddf
    _META_CACHE["counts"] = {
        "questions": int(qdf.shape[0]) if qdf is not None else 0,
        "scales": int(sdf.shape[0]) if sdf is not None else 0,
        "demographics": int(ddf.shape[0]) if ddf is not None else 0,
    }

    # In-memory PS-wide DataFrame (cached)
    df_inmem = preload_pswide_dataframe()
    _INMEM_STATE.update({
        "mode": f"pswide_df({engine})",
        "rows": int(df_inmem.shape[0]) if isinstance(df_inmem, pd.DataFrame) else 0
    })

    # Lift last-engine for UI
    global _LAST_ENGINE
    _LAST_ENGINE = engine

    return {
        "engine": engine,
        "inmem_mode": _INMEM_STATE["mode"],
        "inmem_rows": _INMEM_STATE["rows"],
        "metadata_counts": dict(_META_CACHE["counts"]),
        "pswide_only": True,
        "parquet_dir": PARQUET_ROOTDIR,
        "csv_path": LOCAL_GZ_PATH,
        "agree_src": _AGREE_SRC,
    }

# Back-compat API used by older main.py versions
@st.cache_resource(show_spinner="⚡ Warming up data backend…")
def prewarm_fastpath() -> str:
    summary = prewarm_all()
    return "parquet" if summary.get("engine") == "parquet" else "csv"

def get_backend_info() -> dict:
    # If prewarm_all() hasn't been called yet in this session, call it now (no-op if cached)
    try:
        summary = prewarm_all()
    except Exception:
        summary = {}
    return {
        "last_engine": summary.get("engine", _LAST_ENGINE),
        "inmem_mode": summary.get("inmem_mode", _INMEM_STATE["mode"]),
        "inmem_rows": int(summary.get("inmem_rows", _INMEM_STATE["rows"] or 0)),
        "metadata_counts": summary.get("metadata_counts", _META_CACHE.get("counts", {})),
        "pswide_only": summary.get("pswide_only", True),
        "parquet_dir": PARQUET_ROOTDIR,
        "csv_path": LOCAL_GZ_PATH,
        "agree_src": summary.get("agree_src", _AGREE_SRC),
    }
