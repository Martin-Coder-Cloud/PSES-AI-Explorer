# utils/data_loader.py — Parquet-first loader with CSV fallback + metadata + PS-wide in-memory preload
from __future__ import annotations

import os
import time
from typing import Iterable, Optional

import pandas as pd
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

# Google Drive CSV (set real ID in Streamlit secrets as RESULTS2024_FILE_ID)
GDRIVE_FILE_ID_FALLBACK = ""  # optional placeholder; prefer st.secrets["RESULTS2024_FILE_ID"]
LOCAL_GZ_PATH = os.environ.get("PSES_RESULTS_GZ", "/tmp/Results2024.csv.gz")

# Parquet dataset location (directory). Prefer a persistent folder.
PARQUET_ROOTDIR = os.environ.get("PSES_PARQUET_DIR", "data/parquet/PSES_Results2024")
PARQUET_FLAG = os.path.join(PARQUET_ROOTDIR, "_BUILD_OK")

# Output schema (normalized)
OUT_COLS = [
    "year", "question_code", "group_value", "n",
    "positive_pct", "neutral_pct", "negative_pct",
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
    "answer1": "Float32", "answer2": "Float32", "answer3": "Float32",
    "answer4": "Float32", "answer5": "Float32", "answer6": "Float32", "answer7": "Float32",
}

# Minimal column set to read from CSV (include LEVEL1ID for PS-wide filter)
CSV_USECOLS = [
    "LEVEL1ID",  # may be missing in some exports; handled defensively
    "SURVEYR", "QUESTION", "DEMCODE",
    "ANSCOUNT", "POSITIVE", "NEUTRAL", "NEGATIVE",
    "answer1", "answer2", "answer3", "answer4", "answer5", "answer6", "answer7",
]

# =============================================================================
# Internal diagnostics & compatibility
# =============================================================================
_LAST_DIAG: dict = {}
_LAST_ENGINE: str = "unknown"

# Optional debug notes (you can display them in a Status panel)
_DEBUG_NOTES: list[str] = []

# Back-compat names some older code may read
LAST_BACKEND: str = "unknown"
BACKEND_IN_USE: str = "unknown"

def _note(msg: str) -> None:
    try:
        _DEBUG_NOTES.append(str(msg))
    except Exception:
        pass

def _set_diag(**kwargs):
    _LAST_DIAG.clear()
    _LAST_DIAG.update(kwargs)

def get_last_query_diag() -> dict:
    """
    Returns diagnostics for the most recent load_results2024_filtered call:
      { engine, elapsed_ms, rows, question_code, years, group_value,
        parquet_dir, csv_path, parquet_error }
    """
    return dict(_LAST_DIAG)

def get_last_backend() -> str:
    """Compatibility shim for menu code that probes last backend name."""
    return _LAST_ENGINE or LAST_BACKEND or BACKEND_IN_USE or "unknown"

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
# Build Parquet one-time (filtered to PS-wide if LEVEL1ID exists)
# =============================================================================
def _build_parquet_with_duckdb(csv_path: str) -> None:
    import duckdb
    os.makedirs(PARQUET_ROOTDIR, exist_ok=True)
    con = duckdb.connect()

    # Build SELECT with optional LEVEL1ID==0 filter
    # We keep DEMCODE values (demographic breakdowns) at the PS-wide level.
    con.execute("""
        CREATE OR REPLACE TABLE pses AS
        SELECT
          CAST(SURVEYR AS INT)                                 AS year,
          CAST(QUESTION AS VARCHAR)                            AS question_code,
          COALESCE(NULLIF(TRIM(CAST(DEMCODE AS VARCHAR)), ''),'All') AS group_value,
          CAST(ANSCOUNT AS INT)                                AS n,
          CAST(POSITIVE AS DOUBLE)                             AS positive_pct,
          CAST(NEUTRAL  AS DOUBLE)                             AS neutral_pct,
          CAST(NEGATIVE AS DOUBLE)                             AS negative_pct,
          CAST(answer1  AS DOUBLE) AS answer1,
          CAST(answer2  AS DOUBLE) AS answer2,
          CAST(answer3  AS DOUBLE) AS answer3,
          CAST(answer4  AS DOUBLE) AS answer4,
          CAST(answer5  AS DOUBLE) AS answer5,
          CAST(answer6  AS DOUBLE) AS answer6,
          CAST(answer7  AS DOUBLE) AS answer7
        FROM read_csv_auto(?, header=true)
        WHERE
          -- If LEVEL1ID exists, keep PS-wide only; else include all rows.
          COALESCE(try_cast(LEVEL1ID AS INT), 0) = 0
          OR NOT EXISTS(SELECT 1 FROM (SELECT * FROM read_csv_auto(?, header=true) LIMIT 1) t WHERE TRUE)
    """, [csv_path, csv_path])

    con.execute(f"""
        COPY pses TO '{PARQUET_ROOTDIR}'
        (FORMAT PARQUET, COMPRESSION 'ZSTD', ROW_GROUP_SIZE 1000000,
         PARTITION_BY (year, question_code));
    """)

def _build_parquet_with_pandas(csv_path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.read_csv(csv_path, compression="gzip", usecols=[c for c in CSV_USECOLS if c != "LEVEL1ID"], low_memory=False)
    # Try to bring LEVEL1ID if present even if not in usecols (robustness)
    try:
        cols = pd.read_csv(csv_path, compression="gzip", nrows=0).columns
        if "LEVEL1ID" in cols and "LEVEL1ID" not in df.columns:
            # re-read minimally to fetch LEVEL1ID for filtering
            df2 = pd.read_csv(csv_path, compression="gzip", usecols=["LEVEL1ID"], low_memory=True)
            df["LEVEL1ID"] = df2["LEVEL1ID"]
    except Exception:
        pass

    # Filter to PS-wide if LEVEL1ID is present
    if "LEVEL1ID" in df.columns:
        df = df[pd.to_numeric(df["LEVEL1ID"], errors="coerce").fillna(0).astype(int).eq(0)]

    out = pd.DataFrame({
        "year":          pd.to_numeric(df["SURVEYR"], errors="coerce").astype("Int64"),
        "question_code": df["QUESTION"].astype("string"),
        "group_value":   df["DEMCODE"].astype("string"),
        "n":             pd.to_numeric(df["ANSCOUNT"], errors="coerce").astype("Int64"),
        "positive_pct":  pd.to_numeric(df["POSITIVE"], errors="coerce"),
        "neutral_pct":   pd.to_numeric(df["NEUTRAL"],  errors="coerce"),
        "negative_pct":  pd.to_numeric(df["NEGATIVE"], errors="coerce"),
        "answer1": pd.to_numeric(df.get("answer1"), errors="coerce"),
        "answer2": pd.to_numeric(df.get("answer2"), errors="coerce"),
        "answer3": pd.to_numeric(df.get("answer3"), errors="coerce"),
        "answer4": pd.to_numeric(df.get("answer4"), errors="coerce"),
        "answer5": pd.to_numeric(df.get("answer5"), errors="coerce"),
        "answer6": pd.to_numeric(df.get("answer6"), errors="coerce"),
        "answer7": pd.to_numeric(df.get("answer7"), errors="coerce"),
    })
    # Normalize group_value: empty/NA → "All"
    out["group_value"] = out["group_value"].fillna("All")
    out.loc[out["group_value"].astype("string").str.strip() == "", "group_value"] = "All"

    os.makedirs(PARQUET_ROOTDIR, exist_ok=True)
    table = pa.Table.from_pandas(out[OUT_COLS], preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=PARQUET_ROOTDIR,
        partition_cols=["year", "question_code"],
        compression="zstd",
    )

@st.cache_resource(show_spinner="🗂️ Preparing Parquet dataset (one-time)…")
def ensure_parquet_dataset() -> str:
    """Ensures a partitioned Parquet dataset (PS-wide if LEVEL1ID present) exists and returns its root directory."""
    if not _pyarrow_available():
        raise RuntimeError("pyarrow is required for Parquet fast path.")
    csv_path = ensure_results2024_local()

    if os.path.isdir(PARQUET_ROOTDIR) and os.path.exists(PARQUET_FLAG):
        return PARQUET_ROOTDIR

    os.makedirs(PARQUET_ROOTDIR, exist_ok=True)

    if _duckdb_available():
        _build_parquet_with_duckdb(csv_path)
        _note("Parquet built with DuckDB.")
    else:
        _build_parquet_with_pandas(csv_path)
        _note("Parquet built with pandas/pyarrow.")

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
        _note("Questions metadata not found.")
        return pd.DataFrame(columns=["code", "text", "display"])
    qdf = pd.read_excel(path)
    cols = {c.lower(): c for c in qdf.columns}
    def col(name: str) -> str | None:
        return cols.get(name.lower())
    code_col = col("question")
    text_col = col("English")
    if not code_col or not text_col:
        _note("Questions metadata missing required columns (question/English).")
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
        _note("Scales metadata not found.")
        return pd.DataFrame()
    sdf = pd.read_excel(path)
    sdf.columns = sdf.columns.str.strip().str.lower()
    # Compute a normalized join key for Q-code
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
    else:
        _note("Scales metadata has no code/question column.")
    return sdf

@st.cache_resource(show_spinner=False)
def load_demographics_metadata() -> pd.DataFrame:
    path = _path_candidate("metadata/Demographics.xlsx", "/mnt/data/Demographics.xlsx")
    if not path:
        _note("Demographics metadata not found.")
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
    If Parquet exists, reads the entire Parquet dataset (already LEVEL1ID-filtered during build).
    Else, streams the CSV and filters LEVEL1ID==0.
    """
    try:
        if os.path.isdir(PARQUET_ROOTDIR) and os.path.exists(PARQUET_FLAG) and _pyarrow_available():
            import pyarrow.dataset as ds
            dataset = ds.dataset(PARQUET_ROOTDIR, format="parquet")
            table = dataset.to_table(columns=OUT_COLS)
            df = table.to_pandas(types_mapper=pd.ArrowDtype)
            _note("PS-wide in-memory loaded from Parquet.")
        else:
            # CSV streaming preload
            path = ensure_results2024_local()
            frames: list[pd.DataFrame] = []
            for chunk in pd.read_csv(path, compression="gzip", usecols=CSV_USECOLS, chunksize=1_500_000, low_memory=True):
                # Keep PS-wide rows only if LEVEL1ID exists
                if "LEVEL1ID" in chunk.columns:
                    mask_lvl = pd.to_numeric(chunk["LEVEL1ID"], errors="coerce").fillna(0).astype(int).eq(0)
                    chunk = chunk.loc[mask_lvl, :]
                sel = chunk  # all DEMCODEs preserved at PS-wide
                out = pd.DataFrame({
                    "year":          pd.to_numeric(sel["SURVEYR"], errors="coerce"),
                    "question_code": sel["QUESTION"].astype("string"),
                    "group_value":   sel["DEMCODE"].astype("string").fillna("All"),
                    "n":             pd.to_numeric(sel["ANSCOUNT"], errors="coerce"),
                    "positive_pct":  pd.to_numeric(sel["POSITIVE"], errors="coerce"),
                    "neutral_pct":   pd.to_numeric(sel["NEUTRAL"],  errors="coerce"),
                    "negative_pct":  pd.to_numeric(sel["NEGATIVE"], errors="coerce"),
                    "answer1": pd.to_numeric(sel.get("answer1"), errors="coerce"),
                    "answer2": pd.to_numeric(sel.get("answer2"), errors="coerce"),
                    "answer3": pd.to_numeric(sel.get("answer3"), errors="coerce"),
                    "answer4": pd.to_numeric(sel.get("answer4"), errors="coerce"),
                    "answer5": pd.to_numeric(sel.get("answer5"), errors="coerce"),
                    "answer6": pd.to_numeric(sel.get("answer6"), errors="coerce"),
                    "answer7": pd.to_numeric(sel.get("answer7"), errors="coerce"),
                })
                out.loc[out["group_value"].astype("string").str.strip() == "", "group_value"] = "All"
                frames.append(out)
            df = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUT_COLS))
            _note("PS-wide in-memory loaded from CSV stream.")

        # Cast to friendly dtypes
        if not df.empty:
            df = df.reindex(columns=OUT_COLS)
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(DTYPES["year"])
            df["n"]    = pd.to_numeric(df["n"], errors="coerce").astype(DTYPES["n"])
            for c in ["positive_pct","neutral_pct","negative_pct",
                      "answer1","answer2","answer3","answer4","answer5","answer6","answer7"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(DTYPES[c])
            df["question_code"] = df["question_code"].astype(DTYPES["question_code"])
            df["group_value"]   = df["group_value"].astype(DTYPES["group_value"])
        return df
    except Exception as e:
        _note(f"PS-wide preload failed: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=OUT_COLS)

# Track in-memory status
_INMEM_STATE: dict = {"mode": "none", "rows": 0}
_META_CACHE: dict = {"counts": {}, "questions": None, "scales": None, "demographics": None}
_PSWIDE_ONLY = True  # this app is PS-wide only by design

# =============================================================================
# Parquet query
# =============================================================================
def _parquet_query(question_code: str, years: Iterable[int | str], group_value: Optional[str]) -> pd.DataFrame:
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    root = ensure_parquet_dataset()
    dataset = ds.dataset(root, format="parquet")

    q = str(question_code).strip()
    years_int = [int(y) for y in years]
    gv = None if (group_value is None or str(group_value).strip() == "" or str(group_value).strip().lower() == "all") else str(group_value).strip()

    filt = (pc.field("question_code") == q) & (pc.field("year").isin(years_int))
    if gv is None:
        filt = filt & (pc.field("group_value") == "All")
    else:
        filt = filt & (pc.field("group_value") == gv)

    tbl = dataset.to_table(columns=OUT_COLS, filter=filt)
    df = tbl.to_pandas(types_mapper=pd.ArrowDtype)

    # Cast to friendly dtypes
    if not df.empty:
        df = df.reindex(columns=OUT_COLS)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(DTYPES["year"])
        df["n"]    = pd.to_numeric(df["n"], errors="coerce").astype(DTYPES["n"])
        for c in ["positive_pct","neutral_pct","negative_pct",
                  "answer1","answer2","answer3","answer4","answer5","answer6","answer7"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(DTYPES[c])
        df["question_code"] = df["question_code"].astype(DTYPES["question_code"])
        df["group_value"]   = df["group_value"].astype(DTYPES["group_value"])
    return df

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
    gv = None if (group_value is None or str(group_value).strip() == "" or str(group_value).strip().lower() == "all") else str(group_value).strip()

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, compression="gzip", usecols=CSV_USECOLS, chunksize=chunksize, low_memory=True):
        # PS-wide only if LEVEL1ID exists
        if "LEVEL1ID" in chunk.columns:
            mask_lvl = pd.to_numeric(chunk["LEVEL1ID"], errors="coerce").fillna(0).astype(int).eq(0)
        else:
            mask_lvl = pd.Series(True, index=chunk.index)

        mask = (chunk["QUESTION"].astype(str) == question_code) & \
               (pd.to_numeric(chunk["SURVEYR"], errors="coerce").isin(years_int)) & \
               mask_lvl

        if gv is None:
            gv_ser = chunk["DEMCODE"].astype(str).str.strip()
            mask &= (gv_ser.eq("")) | (gv_ser.isna())
        else:
            mask &= (chunk["DEMCODE"].astype(str).str.strip() == gv)

        if mask.any():
            sel = chunk.loc[mask, :]
            out = pd.DataFrame({
                "year":          pd.to_numeric(sel["SURVEYR"], errors="coerce"),
                "question_code": sel["QUESTION"].astype("string"),
                "group_value":   sel["DEMCODE"].astype("string").fillna("All"),
                "n":             pd.to_numeric(sel["ANSCOUNT"], errors="coerce"),
                "positive_pct":  pd.to_numeric(sel["POSITIVE"], errors="coerce"),
                "neutral_pct":   pd.to_numeric(sel["NEUTRAL"],  errors="coerce"),
                "negative_pct":  pd.to_numeric(sel["NEGATIVE"], errors="coerce"),
                "answer1": pd.to_numeric(sel.get("answer1"), errors="coerce"),
                "answer2": pd.to_numeric(sel.get("answer2"), errors="coerce"),
                "answer3": pd.to_numeric(sel.get("answer3"), errors="coerce"),
                "answer4": pd.to_numeric(sel.get("answer4"), errors="coerce"),
                "answer5": pd.to_numeric(sel.get("answer5"), errors="coerce"),
                "answer6": pd.to_numeric(sel.get("answer6"), errors="coerce"),
                "answer7": pd.to_numeric(sel.get("answer7"), errors="coerce"),
            })
            frames.append(out)

    if not frames:
        return pd.DataFrame(columns=OUT_COLS)

    df = pd.concat(frames, ignore_index=True)
    if not df.empty:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(DTYPES["year"])
        df["n"]    = pd.to_numeric(df["n"], errors="coerce").astype(DTYPES["n"])
        for c in ["positive_pct","neutral_pct","negative_pct",
                  "answer1","answer2","answer3","answer4","answer5","answer6","answer7"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(DTYPES[c])
        df["question_code"] = df["question_code"].astype(DTYPES["question_code"])
        df["group_value"]   = df["group_value"].astype(DTYPES["group_value"])
    return df[OUT_COLS]

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
            q = str(question_code).strip()
            years_int = [int(y) for y in years]
            gv_norm = None if (group_value is None or str(group_value).strip() == "" or str(group_value).strip().lower() == "all") else str(group_value).strip()
            df = df_all[(df_all["question_code"] == q) & (df_all["year"].astype(int).isin(years_int))]
            if gv_norm is None:
                df = df[df["group_value"] == "All"]
            else:
                df = df[df["group_value"] == gv_norm]
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
            )
            return df
    except Exception as e:
        _note(f"In-memory filter failed: {type(e).__name__}: {e}")
        # fall through to Parquet/CSV

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
                group_value=("All" if group_value in (None, "", "all", "All") else str(group_value)),
                parquet_dir=PARQUET_ROOTDIR,
                csv_path=LOCAL_GZ_PATH,
                parquet_error=None,
            )
            return df
        except Exception as e:
            parquet_error = str(e)
            _note(f"Parquet query failed: {parquet_error}")

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
        group_value=("All" if group_value in (None, "", "all", "All") else str(group_value)),
        parquet_dir=PARQUET_ROOTDIR,
        csv_path=LOCAL_GZ_PATH,
        parquet_error=parquet_error,
    )
    return df

# =============================================================================
# Helpers for UI (Diagnostics / prewarm)
# =============================================================================
def _compute_pswide_rowcount_parquet() -> int:
    try:
        import pyarrow.dataset as ds
        root = ensure_parquet_dataset()
        dataset = ds.dataset(root, format="parquet")
        # dataset already filtered to LEVEL1ID==0 at build; count all rows
        return dataset.count_rows()
    except Exception:
        return 0

def _compute_pswide_rowcount_csv() -> int:
    try:
        path = ensure_results2024_local()
        rows = 0
        for chunk in pd.read_csv(
            path, compression="gzip", usecols=CSV_USECOLS, chunksize=2_000_000, low_memory=True
        ):
            if "LEVEL1ID" in chunk.columns:
                mask = pd.to_numeric(chunk["LEVEL1ID"], errors="coerce").fillna(0).astype(int).eq(0)
            else:
                # Fallback heuristic (no LEVEL1ID): treat blank DEMCODE "All" rows only
                gv = chunk["DEMCODE"].astype(str).str.strip()
                mask = (gv.eq("")) | (gv.isna())
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
    _DEBUG_NOTES.clear()

    # Ensure raw results are present
    ensure_results2024_local()
    _note(f"CSV ready at {LOCAL_GZ_PATH}")

    # Prefer Parquet if available (and build if needed)
    engine = "csv"
    try:
        ensure_parquet_dataset()
        engine = "parquet"
        _note(f"Parquet dataset ready at {PARQUET_ROOTDIR}")
    except Exception as e:
        engine = "csv"
        _note(f"Parquet unavailable, using CSV: {type(e).__name__}: {e}")

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
    _note(f"Metadata counts — Q={_META_CACHE['counts']['questions']}, "
          f"Scales={_META_CACHE['counts']['scales']}, Demos={_META_CACHE['counts']['demographics']}")

    # In-memory PS-wide DataFrame (cached)
    df_inmem = preload_pswide_dataframe()
    _INMEM_STATE.update({
        "mode": f"pswide_df({engine})",
        "rows": int(df_inmem.shape[0]) if isinstance(df_inmem, pd.DataFrame) else 0
    })
    _note(f"In-memory rows: {_INMEM_STATE['rows']:,}")

    # Lift last-engine for UI/compat
    global _LAST_ENGINE, LAST_BACKEND, BACKEND_IN_USE
    _LAST_ENGINE = engine
    LAST_BACKEND = engine
    BACKEND_IN_USE = engine

    return {
        "engine": engine,
        "inmem_mode": _INMEM_STATE["mode"],
        "inmem_rows": _INMEM_STATE["rows"],
        "metadata_counts": dict(_META_CACHE["counts"]),
        "pswide_only": True,
        "parquet_dir": PARQUET_ROOTDIR,
        "csv_path": LOCAL_GZ_PATH,
        "debug": {"notes": list(_DEBUG_NOTES)},
    }

# Back-compat API used by older main.py versions
@st.cache_resource(show_spinner="⚡ Warming up data backend…")
def prewarm_fastpath() -> str:
    """
    Ensure CSV is present and Parquet dataset is built (one-time) and PS-wide is preloaded.
    Returns 'parquet' if Parquet is ready, else 'csv'.
    """
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
        "debug": summary.get("debug", {"notes": list(_DEBUG_NOTES)}),
    }
