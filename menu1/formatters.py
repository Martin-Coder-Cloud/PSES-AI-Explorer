# app/menu1/formatters.py
"""
Formatting and normalization helpers for Menu 1 results.
These functions prepare raw query outputs into clean, display-ready DataFrames.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd

# ---------------------------------------------------------------------------
# Suppression handling
# ---------------------------------------------------------------------------
def drop_suppressed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove values coded as 999 or 9999 (suppression / N/A).
    Works across common numeric columns (answers, POS/NEU/NEG, counts).
    """
    if df.empty:
        return df

    out = df.copy()
    cols = [f"answer{i}" for i in range(1, 8)] + [
        "POSITIVE", "NEUTRAL", "NEGATIVE",
        "ANSCOUNT", "positive_pct", "neutral_pct", "negative_pct", "agree_pct", "n"
    ]

    for c in cols:
        if c in out.columns:
            v = pd.to_numeric(out[c], errors="coerce")
            out.loc[v.isin([999, 9999]), c] = pd.NA

    return out

# ---------------------------------------------------------------------------
# Scale labels
# ---------------------------------------------------------------------------
def scale_pairs(scales_df: pd.DataFrame, question_code: str) -> List[Tuple[str, str]]:
    """
    For a given question code, return list of (column, label) for answer1–answer7.
    Falls back to generic "Answer i" when metadata is missing.
    """
    sdf = scales_df.copy()
    candidates = pd.DataFrame()

    for key in ["code", "question"]:
        if key in sdf.columns:
            candidates = sdf[sdf[key].astype(str).str.upper() == str(question_code).upper()]
            if not candidates.empty:
                break

    pairs = []
    for i in range(1, 8):
        col = f"answer{i}"
        lbl = None
        if not candidates.empty and col in candidates.columns:
            vals = candidates[col].dropna().astype(str)
            if not vals.empty:
                lbl = vals.iloc[0].strip()
        pairs.append((col, lbl or f"Answer {i}"))

    return pairs

# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------
def format_display(
    df_slice: pd.DataFrame,
    dem_disp_map: Dict,
    category_in_play: bool,
    scale_pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Format a filtered slice of results into a clean, user-facing table.
    - Adds Year and Demographic labels
    - Renames columns using scale_pairs
    - Rounds percentages
    """
    if df_slice.empty:
        return df_slice.copy()

    out = df_slice.copy()

    # Year column
    out["YearNum"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["Year"] = out["YearNum"].astype(str)

    # Demographic label (if applicable)
    if category_in_play:
        def to_label(code):
            if code is None or (isinstance(code, float) and pd.isna(code)) or str(code).strip() == "":
                return "All respondents"
            return dem_disp_map.get(code, dem_disp_map.get(str(code), str(code)))
        out["Demographic"] = out["group_value"].apply(to_label)

    # Map answer labels
    dist_cols = [k for k, _ in scale_pairs if k in out.columns]
    rename_map = {k: v for k, v in scale_pairs if k in out.columns}

    keep_cols = (
        ["YearNum", "Year"]
        + (["Demographic"] if category_in_play else [])
        + dist_cols
        + ["positive_pct", "neutral_pct", "negative_pct", "agree_pct", "n"]
    )
    keep_cols = [c for c in keep_cols if c in out.columns]

    out = out[keep_cols].rename(columns=rename_map).copy()
    out = out.rename(columns={
        "positive_pct": "Positive",
        "neutral_pct": "Neutral",
        "negative_pct": "Negative",
        "agree_pct": "Agree",
    })

    # Sort: Year desc, Demographic asc
    sort_cols = ["YearNum"] + (["Demographic"] if category_in_play else [])
    sort_asc = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    out = out.drop(columns=["YearNum"])

    # Numerics: round all but Year/Demographic
    for c in out.columns:
        if c not in ("Year", "Demographic"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    pct_like = [c for c in out.columns if c not in ("Year", "Demographic", "n")]
    if pct_like:
        out[pct_like] = out[pct_like].round(1)
    if "n" in out.columns:
        out["n"] = pd.to_numeric(out["n"], errors="coerce").astype("Int64")

    return out

# ---------------------------------------------------------------------------
# Metric detection
# ---------------------------------------------------------------------------
def detect_metric(df_disp: pd.DataFrame, scale_pairs: List[Tuple[str, str]]) -> Dict[str, str]:
    """
    Decide which metric column to use for summary tabulation:
    - Prefer Positive (% positive)
    - Else % agree
    - Else the first available answer label
    Returns dict: {"metric_col": colname, "metric_label": label}
    """
    if df_disp.empty:
        return {"metric_col": "Positive", "metric_label": "% positive"}

    cols_l = {c.lower(): c for c in df_disp.columns}

    if "positive" in cols_l and pd.to_numeric(df_disp[cols_l["positive"]], errors="coerce").notna().any():
        return {"metric_col": cols_l["positive"], "metric_label": "% positive"}
    if "agree" in cols_l and pd.to_numeric(df_disp[cols_l["agree"]], errors="coerce").notna().any():
        return {"metric_col": cols_l["agree"], "metric_label": "% agree"}

    for _, label in scale_pairs:
        if label and label in df_disp.columns and pd.to_numeric(df_disp[label], errors="coerce").notna().any():
            return {"metric_col": label, "metric_label": f"% {label}"}

    return {"metric_col": "Positive", "metric_label": "% positive"}
