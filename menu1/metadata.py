# app/menu1/metadata.py
"""
Cached metadata loaders for Menu 1.
Wraps Excel sheets and returns normalized DataFrames.
"""

from __future__ import annotations
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_questions(path: str = "metadata/Survey Questions.xlsx") -> pd.DataFrame:
    """
    Load survey questions metadata.
    Returns DataFrame with columns: code, text, display
    """
    qdf = pd.read_excel(path)
    qdf.columns = [c.strip().lower() for c in qdf.columns]

    # Normalize column names
    if "question" in qdf.columns and "english" in qdf.columns:
        qdf = qdf.rename(columns={"question": "code", "english": "text"})
    elif "code" not in qdf.columns or "text" not in qdf.columns:
        raise ValueError("Survey Questions.xlsx missing required columns")

    qdf["code"] = qdf["code"].astype(str)

    # For sorting by Q number (Q01, Q02, …)
    qdf["qnum"] = qdf["code"].str.extract(r"Q?(\d+)", expand=False)
    with pd.option_context("mode.chained_assignment", None):
        qdf["qnum"] = pd.to_numeric(qdf["qnum"], errors="coerce")

    qdf = qdf.sort_values(["qnum", "code"], na_position="last")

    # Display string (code – text)
    qdf["display"] = qdf["code"].astype(str) + " – " + qdf["text"].astype(str)

    return qdf[["code", "text", "display"]].copy()

# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_scales(path: str = "metadata/Survey Scales.xlsx") -> pd.DataFrame:
    """
    Load survey scales metadata.
    Returns DataFrame with lower-cased column names.
    """
    sdf = pd.read_excel(path)
    sdf.columns = [c.strip().lower() for c in sdf.columns]
    return sdf.copy()

# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_demographics(path: str = "metadata/Demographics.xlsx") -> pd.DataFrame:
    """
    Load demographics metadata.
    Returns DataFrame with stripped column names.
    """
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df.copy()
