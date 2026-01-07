# utils/data_loader.py — SAFE loader for Streamlit Cloud (no heavy prewarm)
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

# Google Drive file ID (fallback if Streamlit secrets are missing/reset)
GDRIVE_FILE_ID_FALLBACK = "1VdMQQfEP-BNXle8GeD-Z_upt2pPIGvc8"

# Where we store the downloaded gzip on Streamlit Cloud
LOCAL_GZ_PATH = os.environ.get("PSES_RESULTS_GZ", "/tmp/Results2024.csv.gz")

# Optional Parquet fast path (DISABLED by default for stability on Streamlit Cloud)
PSES_ENABLE_PARQUET = os.environ.get("PSES_ENABLE_PARQUET", "0").strip().lower() in ("1", "true", "yes")
PARQUET_ROOTDIR = os.environ.get("PSES_PARQUET_DIR", "/tmp/pses_parquet/PSES_Results2024_v2")
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

# Minimal columns to read
CSV_USECOLS = [
    "LEVEL1ID",
    "SURVEYR", "QUESTION", "DEMCODE",
    "ANSCOUNT", "POSITIVE", "NEUTRAL", "NEGATIVE", "AGREE",
    "answer1", "answer2", "answer3", "answer4", "answer5", "answer6", "answer7",
]

# =============================================================================
# Diagnostics
# =============================================================================
_LAST_DIAG: dict = {}
_LAST_ENGINE: str = "unknown"
_AGREE_SRC: Optional[str] = None

def _set_diag(**kwargs):
    _LAST_DIAG.clear()
    _LAST_DIAG.update(kwargs)

def get_last_query_diag() -> dict:
    return dict(_LAST_DIAG)

# Lightweight caches (metadata only)
_META_CACHE: dict = {"counts": {}, "questions": None, "scales": None, "demographics": None}

# =============================================================================
# Capability checks
# =============================================================================
def _pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.dataset as ds  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
        return True
    except Exception:
        return False

def _duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401
        return True
    except Exception:
        return False

# =============================================================================
# CSV presence (cached)
# =============================================================================
@st.cache_resource(show_spinner="📥 Downloading Results2024.csv.gz…")
def ensure_results2024_local(file_id: Optional[str] = None) -> str:
    try:
        import gdown  # noqa: F401
    except Exception as e:
        raise RuntimeError("Missing dependency: gdown (must be in requirements.txt).") from e

    fid = file_id or st.secrets.get("RESULTS2024_FILE_ID") or GDRIVE_FILE_ID_FALLBACK
    if not fid:
        raise RuntimeError(
            "RESULTS2024_FILE_ID is missing. Set Streamlit Secrets RESULTS2024_FILE_ID "
            "or set GDRIVE_FILE_ID_FALLBACK in utils/data_loader.py."
        )

    if os.path.exists(LOCAL_GZ_PATH) and os.path.getsize(LOCAL_GZ_PATH) > 0:
        return LOCAL_GZ_PATH

    parent = os.path.dirname(LOCAL_GZ_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)

    url = f"https://drive.google.com/uc?id={fid}"
    import gdown
    gdown.download(url, LOCAL_GZ_PATH, quiet=False)

    if not os.path.exists(LOCAL_GZ_PATH) or os.path.getsize(LOCAL_GZ_PATH) == 0:
        raise RuntimeError("Download failed or produced an empty file.")
    return LOCAL_GZ_PATH

# =============================================================================
# Agree header detection
# =============================================================================
def _detect_agree_header(csv_path: str) -> Optional[str]:
    try:
        cols = pd.read_csv(csv_path, compression="gzip", nrows=0).columns.tolist()
    except Exception:
        return None
    lower_map = {c.lower(): c for c in cols}
    for key in ["agree", "agree_pct", "agree pct", "agreepercent", "pct_agree"]:
        if key in lower_map:
            return lower_map[key]
    return None

def _with_agree_in_usecols(base: Sequence[str], csv_path: str) -> list[str]:
    agree_col = _detect_agree_header(csv_path)
    global _AGREE_SRC
    _AGREE_SRC = agree_col
    s = {c for c in base}
    if agree_col:
        s.add(agree_col)
    return list(s)

# =============================================================================
# Canonicalization helpers
# =============================================================================
def _canon_demcode_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    s = s.str.replace(r"(\.\d*?[1-9])0+$", r"\1", regex=True)
    s = s.str.replace(r"\.$", "", regex=True)
    s = s.mask(s.isna() | (s == ""), "All")
    return s

def _canon_demcode_value(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "all":
        return None
    import re
    s = re.sub(r"\.0+$", "", s)
    s = re.sub(r"(\.\d*?[1-9])0+$", r"\1", s)
    s = re.sub(r"\.$", "", s)
    return s

# =============================================================================
# Type normalization
# =============================================================================
def _normalize_df_types(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame(columns=OUT_COLS)

    df = df.reindex(columns=OUT_COLS)

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(DTYPES["year"])
    df["n"] = pd.to_numeric(df["n"], errors="coerce").astype(DTYPES["n"])
    for c in ["positive_pct", "neutral_pct", "negative_pct", "agree_pct",
              "answer1", "answer2", "answer3", "answer4", "answer5", "answer6", "answer7"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(DTYPES[c])

    df["question_code"] = df["question_code"].astype("string").str.strip().str.upper().astype(DTYPES["question_code"])
    df["group_value"] = _canon_demcode_series(df["group_value"]).astype(DTYPES["group_value"])
    return df

# =============================================================================
# Optional Parquet (lazy, NOT used in prewarm)
# =============================================================================
def _parquet_rowcount(root: str) -> int:
    try:
        import pyarrow.dataset as ds
        dataset = ds.dataset(root, format="parquet")
        return int(dataset.count_rows())
    except Exception:
        return 0

def _build_parquet(csv_path: str) -> None:
    # WARNING: can be heavy on Streamlit Cloud
    os.makedirs(PARQUET_ROOTDIR, exist_ok=True)

    agree_col = _detect_agree_header(csv_path)
    global _AGREE_SRC
    _AGREE_SRC = agree_col

    if _duckdb_available():
        import duckdb
        con = duckdb.connect()

        agree_select = f'CAST("{agree_col}" AS DOUBLE) AS agree_pct,' if agree_col else "CAST(NULL AS DOUBLE) AS agree_pct,"

        con.execute(f"""
            CREATE OR REPLACE TABLE pses AS
            SELECT
              CAST(SURVEYR AS INT)                                   AS year,
              UPPER(TRIM(CAST(QUESTION AS VARCHAR)))                 AS question_code,
              COALESCE(
                NULLIF(
                  REGEXP_REPLACE(
                    REGEXP_REPLACE(TRIM(CAST(DEMCODE AS VARCHAR)), '(\\.[0-9]*?[1-9])0+$', '\\1'),
                    '\\.0+$', ''
                  ),
                  ''
                ),
                'All'
              )                                                       AS group_value,
              CAST(ANSCOUNT AS INT)                                  AS n,
              CAST(POSITIVE AS DOUBLE)                               AS positive_pct,
              CAST(NEUTRAL  AS DOUBLE)                               AS neutral_pct,
              CAST(NEGATIVE AS DOUBLE)                               AS negative_pct,
              {agree_select}
              CAST(answer1 AS DOUBLE) AS answer1,
              CAST(answer2 AS DOUBLE) AS answer2,
              CAST(answer3 AS DOUBLE) AS answer3,
              CAST(answer4 AS DOUBLE) AS answer4,
              CAST(answer5 AS DOUBLE) AS answer5,
              CAST(answer6 AS DOUBLE) AS answer6,
              CAST(answer7 AS DOUBLE) AS answer7
            FROM read_csv_auto(?, header=true)
            WHERE CAST(LEVEL1ID AS INT) = 0
        """, [csv_path])

        con.execute(f"""
            COPY pses TO '{PARQUET_ROOTDIR}'
            (FORMAT PARQUET, COMPRESSION 'ZSTD', ROW_GROUP_SIZE 1000000,
             PARTITION_BY (year, question_code));
        """)
    else:
        # pandas+pyarrow fallback
        import pyarrow as pa
        import pyarrow.parquet as pq

        usecols = _with_agree_in_usecols(CSV_USECOLS, csv_path)
        df = pd.read_csv(csv_path, compression="gzip", usecols=usecols, low_memory=False)

        if "LEVEL1ID" in df.columns:
            df = df[pd.to_numeric(df["LEVEL1ID"], errors="coerce").fillna(0).astype(int).eq(0)]

        out = pd.DataFrame({
            "year": pd.to_numeric(df["SURVEYR"], errors="coerce"),
            "question_code": df["QUESTION"].astype("string").str.strip().str.upper(),
            "group_value": _canon_demcode_series(df["DEMCODE"]),
            "n": pd.to_numeric(df["ANSCOUNT"], errors="coerce"),
            "positive_pct": pd.to_numeric(df["POSITIVE"], errors="coerce"),
            "neutral_pct": pd.to_numeric(df["NEUTRAL"], errors="coerce"),
            "negative_pct": pd.to_numeric(df["NEGATIVE"], errors="coerce"),
            "agree_pct": pd.to_numeric(df.get(agree_col) if agree_col else None, errors="coerce"),
            "answer1": pd.to_numeric(df.get("answer1"), errors="coerce"),
            "answer2": pd.to_numeric(df.get("answer2"), errors="coerce"),
            "answer3": pd.to_numeric(df.get("answer3"), errors="coerce"),
            "answer4": pd.to_numeric(df.get("answer4"), errors="coerce"),
            "answer5": pd.to_numeric(df.get("answer5"), errors="coerce"),
            "answer6": pd.to_numeric(df.get("answer6"), errors="coerce"),
            "answer7": pd.to_numeric(df.get("answer7"), errors="coerce"),
        })

        table = pa.Table.from_pandas(out[OUT_COLS], preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=PARQUET_ROOTDIR,
            partition_cols=["year", "question_code"],
            compression="zstd",
        )

@st.cache_resource(show_spinner="🗂️ Preparing Parquet dataset (one-time)…")
def ensure_parquet_dataset() -> str:
    if not PSES_ENABLE_PARQUET:
        raise RuntimeError("Parquet disabled (set PSES_ENABLE_PARQUET=1 to enable).")
    if not _pyarrow_available():
        raise RuntimeError("pyarrow not available; cannot enable Parquet.")

    csv_path = ensure_results2024_local()

    if os.path.isdir(PARQUET_ROOTDIR) and os.path.exists(PARQUET_FLAG):
        if _parquet_rowcount(PARQUET_ROOTDIR) > 0:
            return PARQUET_ROOTDIR
        shutil.rmtree(PARQUET_ROOTDIR, ignore_errors=True)

    os.makedirs(PARQUET_ROOTDIR, exist_ok=True)
    _build_parquet(csv_path)

    if _parquet_rowcount(PARQUET_ROOTDIR) <= 0:
        raise RuntimeError("Parquet built but contains 0 rows.")

    with open(PARQUET_FLAG, "w") as f:
        f.write("ok")
    return PARQUET_ROOTDIR

def _parquet_query(question_code: str, years: Iterable[int | str], group_value: Optional[str]) -> pd.DataFrame:
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    root = ensure_parquet_dataset()
    dataset = ds.dataset(root, format="parquet")

    q = str(question_code).strip().upper()
    years_int = [int(y) for y in years]
    gv_norm = _canon_demcode_value(group_value)

    filt = (pc.field("question_code") == q) & (pc.field("year").isin(years_int))
    filt = filt & ((pc.field("group_value") == "All") if gv_norm is None else (pc.field("group_value") == gv_norm))

    tbl = dataset.to_table(columns=OUT_COLS, filter=filt)
    df = tbl.to_pandas(types_mapper=pd.ArrowDtype)
    return _normalize_df_types(df)

# =============================================================================
# CSV streaming query (SAFE)
# =============================================================================
def _csv_stream_filter(
    question_code: str,
    years: Iterable[int | str],
    group_value: Optional[str],
    chunksize: int = 1_250_000,
) -> pd.DataFrame:
    path = ensure_results2024_local()
    years_int = [int(y) for y in years]
    q_norm = str(question_code).strip().upper()
    gv_target = _canon_demcode_value(group_value)

    frames: list[pd.DataFrame] = []
    usecols = _with_agree_in_usecols(CSV_USECOLS, path)

    for chunk in pd.read_csv(path, compression="gzip", usecols=usecols, chunksize=chunksize, low_memory=True):
        if "LEVEL1ID" in chunk.columns:
            mask_lvl = pd.to_numeric(chunk["LEVEL1ID"], errors="coerce").fillna(0).astype(int).eq(0)
        else:
            mask_lvl = pd.Series(True, index=chunk.index)

        q_ser = chunk["QUESTION"].astype(str).str.strip().str.upper()
        y_ser = pd.to_numeric(chunk["SURVEYR"], errors="coerce")

        mask = (q_ser == q_norm) & (y_ser.isin(years_int)) & mask_lvl

        gv_ser = _canon_demcode_series(chunk["DEMCODE"])
        mask &= ((gv_ser == "All") if gv_target is None else (gv_ser == gv_target))

        if mask.any():
            sel = chunk.loc[mask, :]
            agree_col = _AGREE_SRC
            out = pd.DataFrame({
                "year": pd.to_numeric(sel["SURVEYR"], errors="coerce"),
                "question_code": sel["QUESTION"].astype("string").str.strip().str.upper(),
                "group_value": _canon_demcode_series(sel["DEMCODE"]),
                "n": pd.to_numeric(sel["ANSCOUNT"], errors="coerce"),
                "positive_pct": pd.to_numeric(sel["POSITIVE"], errors="coerce"),
                "neutral_pct": pd.to_numeric(sel["NEUTRAL"], errors="coerce"),
                "negative_pct": pd.to_numeric(sel["NEGATIVE"], errors="coerce"),
                "agree_pct": pd.to_numeric(sel.get(agree_col) if agree_col else None, errors="coerce"),
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
# Public API
# =============================================================================
@st.cache_data(show_spinner="🔎 Filtering results…")
def load_results2024_filtered(
    question_code: str,
    years: Iterable[int | str],
    group_value: Optional[str] = None,
) -> pd.DataFrame:
    global _LAST_ENGINE
    t0 = time.perf_counter()
    parquet_error = None

    # If Parquet is enabled, try it first (still lazy)
    if PSES_ENABLE_PARQUET and _pyarrow_available():
        try:
            df = _parquet_query(question_code, years, group_value)
            _LAST_ENGINE = "parquet"
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            gv_norm = _canon_demcode_value(group_value)
            _set_diag(
                engine=_LAST_ENGINE,
                elapsed_ms=elapsed_ms,
                rows=int(df.shape[0]),
                question_code=str(question_code),
                years=",".join(str(y) for y in years),
                group_value=("All" if gv_norm is None else gv_norm),
                parquet_dir=PARQUET_ROOTDIR,
                csv_path=LOCAL_GZ_PATH,
                parquet_error=None,
                agree_src=_AGREE_SRC,
            )
            return df
        except Exception as e:
            parquet_error = f"{type(e).__name__}: {e}"

    # Safe default: CSV streaming
    df = _csv_stream_filter(question_code, years, group_value)
    _LAST_ENGINE = "csv"
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    gv_norm = _canon_demcode_value(group_value)
    _set_diag(
        engine=_LAST_ENGINE,
        elapsed_ms=elapsed_ms,
        rows=int(df.shape[0]),
        question_code=str(question_code),
        years=",".join(str(y) for y in years),
        group_value=("All" if gv_norm is None else gv_norm),
        parquet_dir=PARQUET_ROOTDIR,
        csv_path=LOCAL_GZ_PATH,
        parquet_error=parquet_error,
        agree_src=_AGREE_SRC,
    )
    return df

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
    path = _path_candidate("metadata/Survey Questions.xlsx", "/mnt/data/Survey Questions.xlsx")
    if not path:
        return pd.DataFrame(columns=["code", "text", "display"])
    qdf = pd.read_excel(path)
    cols = {c.lower(): c for c in qdf.columns}
    code_col = cols.get("question")
    text_col = cols.get("english")
    if not code_col or not text_col:
        return pd.DataFrame(columns=["code", "text", "display"])
    out = pd.DataFrame({"code": qdf[code_col].astype(str).str.strip(),
                        "text": qdf[text_col].astype(str)})
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
# SAFE prewarm (download + metadata only)
# =============================================================================
@st.cache_resource(show_spinner="⚡ Preparing metadata (light)…")
def prewarm_all() -> dict:
    csv_path = ensure_results2024_local()

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

    return {
        "engine": "csv",
        "parquet_enabled": PSES_ENABLE_PARQUET,
        "metadata_counts": dict(_META_CACHE["counts"]),
        "csv_path": csv_path,
        "agree_src": _AGREE_SRC,
    }

@st.cache_resource(show_spinner="⚡ Warming up data backend…")
def prewarm_fastpath() -> str:
    prewarm_all()
    return "csv"

def get_backend_info() -> dict:
    try:
        summary = prewarm_all()
    except Exception:
        summary = {}
    return {
        "last_engine": _LAST_ENGINE,
        "parquet_enabled": bool(summary.get("parquet_enabled", PSES_ENABLE_PARQUET)),
        "metadata_counts": summary.get("metadata_counts", _META_CACHE.get("counts", {})),
        "csv_path": summary.get("csv_path", LOCAL_GZ_PATH),
        "agree_src": summary.get("agree_src", _AGREE_SRC),
    }
