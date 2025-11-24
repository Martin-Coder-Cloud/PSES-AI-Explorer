# app/menu1/queries.py
"""
Query wrappers and normalization for Menu 1.

- Provides a stable, testable surface over utils.data_loader
- Keeps all column name normalization in one place
- Restores the "All respondents" baseline row in demographic tabulations
- Back-compat: re-exports `normalize_results(df)` for callers that import it
"""

from __future__ import annotations
from typing import Iterable, Optional, List
import pandas as pd

# ---- Optional imports from your existing utils layer ------------------------
try:
    from utils.data_loader import load_results2024_filtered  # main query
except Exception:
    load_results2024_filtered = None  # type: ignore


# ---- Column normalization (defensive, does not change values) ---------------
_OUT_COLS = [
    "year", "question_code", "group_value", "n",
    "positive_pct", "neutral_pct", "negative_pct", "agree_pct",
    "answer1", "answer2", "answer3", "answer4", "answer5", "answer6", "answer7",
]

_RENAME_MAP = {
    # identifiers
    "SURVEYR": "year",
    "survey_year": "year",
    "QUESTION": "question_code",
    "question": "question_code",
    "DEMCODE": "group_value",
    "demcode": "group_value",
    "group": "group_value",
    # measures
    "ANSCOUNT": "n",
    "anscount": "n",
    "POSITIVE": "positive_pct",
    "NEUTRAL": "neutral_pct",
    "NEGATIVE": "negative_pct",
    "ANSWER1": "answer1", "ANSWER2": "answer2", "ANSWER3": "answer3",
    "ANSWER4": "answer4", "ANSWER5": "answer5", "ANSWER6": "answer6", "ANSWER7": "answer7",
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure result frames have a consistent schema expected by the downstream
    formatter/render path. This does NOT change values, only column names and order.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=_OUT_COLS)

    # Map/alias known columns
    cols = {c: _RENAME_MAP.get(c, _RENAME_MAP.get(c.strip().upper(), c)) for c in df.columns}
    out = df.rename(columns=cols).copy()

    # Ensure canonical names if present under known aliases
    if "question_code" not in out.columns and "QUESTION" in df.columns:
        out.rename(columns={"QUESTION": "question_code"}, inplace=True)
    if "year" not in out.columns and "SURVEYR" in df.columns:
        out.rename(columns={"SURVEYR": "year"}, inplace=True)
    if "group_value" not in out.columns:
        for c in ("DEMCODE", "group", "group_code"):
            if c in df.columns:
                out.rename(columns={c: "group_value"}, inplace=True)
                break
    if "n" not in out.columns:
        for c in ("ANSCOUNT", "n_responses", "count"):
            if c in df.columns:
                out.rename(columns={c: "n"}, inplace=True)
                break
    if "positive_pct" not in out.columns and "POSITIVE" in df.columns:
        out.rename(columns={"POSITIVE": "positive_pct"}, inplace=True)
    if "neutral_pct" not in out.columns and "NEUTRAL" in df.columns:
        out.rename(columns={"NEUTRAL": "neutral_pct"}, inplace=True)
    if "negative_pct" not in out.columns and "NEGATIVE" in df.columns:
        out.rename(columns={"NEGATIVE": "negative_pct"}, inplace=True)

    # Ensure all expected columns exist (preserve order)
    for c in _OUT_COLS:
        if c not in out.columns:
            out[c] = pd.NA

    out = out[_OUT_COLS]

    # Light-touch string hygiene for identifiers (values themselves already canonical from loader)
    out["question_code"] = out["question_code"].astype("string").str.strip().str.upper()
    out["group_value"] = out["group_value"].astype("string").str.strip()

    # ---- NA-safe numeric normalization (minimal, defensive) ------------------
    # year as nullable integer
    try:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int16")
    except Exception:
        pass

    # counts as nullable integer
    try:
        out["n"] = pd.to_numeric(out["n"], errors="coerce").astype("Int64")
    except Exception:
        # fall back to leaving as-is if cast fails for any unexpected reason
        pass

    # Measures & distributions: numeric + 9999 sentinel -> NA (so render shows "N/A")
    measure_cols = [
        "positive_pct", "neutral_pct", "negative_pct", "agree_pct",
        "answer1", "answer2", "answer3", "answer4", "answer5", "answer6", "answer7",
    ]
    for c in measure_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            # 9999 denotes not available/applicable in your dataset
            out.loc[out[c] == 9999, c] = pd.NA
            # keep as float (NA-friendly); downstream can format as integer if desired

    return out


# ---- Back-compat export -----------------------------------------------------
def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible alias so older code can:
        from menu1.queries import normalize_results
    without breaking. Delegates to the internal normalizer.
    """
    return _normalize_columns(df)


# ---- Public API -------------------------------------------------------------
def fetch_per_question(
    question_code: str,
    years: Iterable[int | str],
    demcodes: Optional[Iterable[Optional[str]]] = None,
    *,
    include_baseline: bool = True,
) -> pd.DataFrame:
    """
    Fetches rows for a single question across selected years and demographic codes.

    Behavior (restored):
      - If a demographic list is provided, include the PS-wide baseline ("All respondents")
        by prepending group_value=None to the request.
      - If no demcodes are provided (i.e., overall selection), fetch baseline only.
      - Deduplicate codes while preserving order: baseline first, then subgroups.

    Returns a normalized DataFrame with the expected Menu 1 schema.
    """
    if load_results2024_filtered is None:
        # Hard failure path kept minimal; downstream will show a friendly message.
        return pd.DataFrame(columns=_OUT_COLS)

    # 1) Build the exact group_value sequence to fetch
    seq: List[Optional[str]] = []

    # include baseline for any demographic request
    if include_baseline:
        seq.append(None)  # None means "overall" to the loader -> returns group_value == "All"

    # append provided codes (if any)
    if demcodes:
        for code in demcodes:
            seq.append(None if code in (None, "", "All") else str(code))
    else:
        # No demcodes means overall-only (baseline already appended if include_baseline)
        pass

    # De-duplicate while preserving order (ensures baseline first, then unique subgroups)
    seen = set()
    group_values: List[Optional[str]] = []
    for v in seq or [None]:
        key = "None" if v is None else str(v)
        if key not in seen:
            seen.add(key)
            group_values.append(v)

    # 2) Execute queries and collect parts
    parts: List[pd.DataFrame] = []
    for gv in group_values:
        try:
            df_part = load_results2024_filtered(
                question_code=question_code,
                years=list(years),
                group_value=gv,  # None -> overall ("All respondents")
            )
            if df_part is not None and not df_part.empty:
                parts.append(df_part)
        except Exception:
            # Robust to any single-slice error: continue
            continue

    if not parts:
        return pd.DataFrame(columns=_OUT_COLS)

    out = pd.concat(parts, ignore_index=True)
    out = _normalize_columns(out)

    # 3) Friendly ordering: baseline first, then subgroup codes; within each, sort by year asc
    try:
        out["__is_baseline__"] = (out["group_value"].astype("string") == "All")
        order_map = {str(v): i for i, v in enumerate([v for v in group_values if v is not None])}
        out["__sub_order__"] = out["group_value"].astype("string").map(order_map).fillna(1e9)
        out = out.sort_values(["__is_baseline__", "__sub_order__", "year"], ascending=[False, True, True])
        out = out.drop(columns=["__is_baseline__", "__sub_order__"])
    except Exception:
        pass

    return out
