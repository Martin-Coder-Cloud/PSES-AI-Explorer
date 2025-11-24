# app/menu1/state.py
"""
Session state management for Menu 1.

This module centralizes:
- default keys and reset behavior
- simple getters/setters for common values
- a tiny "results stash" API so results can be computed in one place
  and rendered elsewhere (centered) without re-running the query
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import streamlit as st

# ---- Keys (single source of truth) ------------------------------------------

LAST_ACTIVE_MENU_KEY = "last_active_menu"

# Core UI keys
K_SELECTED_CODES      = "menu1_selected_codes"     # List[str] of question codes
K_HITS                = "menu1_hits"               # search results (list of dicts)
K_KW_QUERY            = "menu1_kw_query"           # keyword search string
K_MULTI_QUESTIONS     = "menu1_multi_questions"    # multiselect (display strings)
K_AI_TOGGLE           = "menu1_ai_toggle"          # bool
K_DIAG_TOGGLE         = "menu1_show_diag"          # bool
K_SELECT_ALL_YEARS    = "select_all_years"         # bool master toggle
K_DEMO_MAIN           = "demo_main"                # selected demographic category
K_FIND_HITS_BTN       = "menu1_find_hits"          # button key (just to clear on reset)
K_LAST_QUERY_INFO     = "last_query_info"          # dict with timing/engine, shown in diagnostics

# Per-year checkbox keys (kept compatible with the current UI)
YEAR_KEYS = [f"year_{y}" for y in (2024, 2022, 2020, 2019)]

# Subgroup checkbox namespace prefix, and selected-question checkbox prefix
SUBGROUP_PREFIX = "sub_"     # final key is e.g. f"sub_{demo_name_with_underscores}"
SELECTED_PREFIX = "sel_"     # e.g. "sel_Q01"
HIT_PREFIX      = "kwhit_"   # e.g. "kwhit_Q01"

# Results stash keys (so results can render centered, outside the action row)
K_HAS_RESULTS        = "m1_has_results"
K_TAB_LABELS         = "m1_tab_labels"            # List[str] of question codes with results
K_PIVOT              = "m1_pivot"                 # Summary pivot DataFrame
K_PER_Q_DISP         = "m1_per_q_disp"            # Dict[qcode -> DataFrame]
K_PER_Q_METRIC_COL   = "m1_per_q_metric_col"      # Dict[qcode -> str]
K_PER_Q_METRIC_LABEL = "m1_per_q_metric_label"    # Dict[qcode -> str]
K_CODE_TO_TEXT       = "m1_code_to_text"          # Dict[qcode -> qtext]
K_SELECTED_YEARS     = "m1_selected_years"        # List[int]
K_DEMO_SELECTION     = "m1_demo_selection"        # str
K_SUB_SELECTION      = "m1_sub_selection"         # Optional[str]

# ---- Defaults (import-free to avoid circular deps) --------------------------
# If you want to change these globally, do it in constants.py and pass into set_defaults().
DEFAULTS = {
    K_SELECTED_CODES:     [],
    K_HITS:               [],
    K_KW_QUERY:           "",
    K_MULTI_QUESTIONS:    [],
    K_AI_TOGGLE:          True,   # AI ON by default
    K_DIAG_TOGGLE:        False,  # Diagnostics OFF by default
    K_SELECT_ALL_YEARS:   True,
    K_DEMO_MAIN:          "All respondents",
    K_LAST_QUERY_INFO:    None,
    K_HAS_RESULTS:        False,
}

# ---- Internal helpers -------------------------------------------------------

def _delete_keys(prefixes: List[str], exact_keys: Optional[List[str]] = None) -> None:
    """Remove any session_state keys that match the given prefixes or exact names."""
    exact_keys = exact_keys or []
    keys = list(st.session_state.keys())
    for k in keys:
        if k in exact_keys or any(k.startswith(p) for p in prefixes):
            try:
                del st.session_state[k]
            except Exception:
                pass

# ---- Public API -------------------------------------------------------------

def set_last_active_menu(name: str) -> None:
    st.session_state[LAST_ACTIVE_MENU_KEY] = name

def get_last_active_menu(default: Optional[str] = None) -> Optional[str]:
    return st.session_state.get(LAST_ACTIVE_MENU_KEY, default)

def set_defaults(overrides: Optional[Dict[str, Any]] = None) -> None:
    """Idempotently seed default values (used at page entry)."""
    values = {**DEFAULTS, **(overrides or {})}
    for k, v in values.items():
        st.session_state.setdefault(k, v)

def reset_menu1_state() -> None:
    """
    Hard reset of Menu 1 state. Safe to call when switching menus or
    when the user presses "Reset all parameters".
    """
    exact = [
        K_SELECTED_CODES, K_HITS, K_KW_QUERY, K_MULTI_QUESTIONS,
        K_AI_TOGGLE, K_DIAG_TOGGLE, K_SELECT_ALL_YEARS,
        K_DEMO_MAIN, K_FIND_HITS_BTN, K_LAST_QUERY_INFO,
        # results stash
        K_HAS_RESULTS, K_TAB_LABELS, K_PIVOT, K_PER_Q_DISP,
        K_PER_Q_METRIC_COL, K_PER_Q_METRIC_LABEL, K_CODE_TO_TEXT,
        K_SELECTED_YEARS, K_DEMO_SELECTION, K_SUB_SELECTION,
        # year checkboxes
        *YEAR_KEYS,
    ]
    prefixes = [HIT_PREFIX, SELECTED_PREFIX, SUBGROUP_PREFIX]
    _delete_keys(prefixes=prefixes, exact_keys=exact)
    set_defaults()  # re-seed defaults after wipe

# Common getters/setters (kept minimal; add more as needed)

def get_selected_questions() -> List[str]:
    """Returns the list of selected question CODES (e.g., ['Q01','Q12'])."""
    return list(st.session_state.get(K_SELECTED_CODES, []))

def set_selected_questions(codes: List[str]) -> None:
    st.session_state[K_SELECTED_CODES] = list(codes or [])

def get_ai_enabled() -> bool:
    return bool(st.session_state.get(K_AI_TOGGLE, True))

def set_ai_enabled(enabled: bool) -> None:
    st.session_state[K_AI_TOGGLE] = bool(enabled)

def get_diag_enabled() -> bool:
    return bool(st.session_state.get(K_DIAG_TOGGLE, False))

def set_diag_enabled(enabled: bool) -> None:
    st.session_state[K_DIAG_TOGGLE] = bool(enabled)

def set_last_query_info(info: Dict[str, Any]) -> None:
    st.session_state[K_LAST_QUERY_INFO] = info

def get_last_query_info(default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    return st.session_state.get(K_LAST_QUERY_INFO, default)

# ---- Results stash API ------------------------------------------------------

def clear_results() -> None:
    st.session_state[K_HAS_RESULTS] = False
    for k in (
        K_TAB_LABELS, K_PIVOT, K_PER_Q_DISP, K_PER_Q_METRIC_COL,
        K_PER_Q_METRIC_LABEL, K_CODE_TO_TEXT, K_SELECTED_YEARS,
        K_DEMO_SELECTION, K_SUB_SELECTION
    ):
        if k in st.session_state:
            try:
                del st.session_state[k]
            except Exception:
                pass

def stash_results(payload: Dict[str, Any]) -> None:
    """
    Save computed results so rendering can happen elsewhere (and centered).
    Expected keys in payload:
      - 'per_q_disp': Dict[str, DataFrame]
      - 'per_q_metric_col': Dict[str, str]
      - 'per_q_metric_label': Dict[str, str]
      - 'pivot': DataFrame
      - 'tab_labels': List[str]
      - 'years': List[int]
      - 'demo_selection': str
      - 'sub_selection': Optional[str]
      - 'code_to_text': Dict[str, str]   (optional but useful)
    """
    clear_results()
    st.session_state[K_PER_Q_DISP]         = payload.get("per_q_disp")
    st.session_state[K_PER_Q_METRIC_COL]   = payload.get("per_q_metric_col")
    st.session_state[K_PER_Q_METRIC_LABEL] = payload.get("per_q_metric_label")
    st.session_state[K_PIVOT]              = payload.get("pivot")
    st.session_state[K_TAB_LABELS]         = payload.get("tab_labels", [])
    st.session_state[K_SELECTED_YEARS]     = payload.get("years", [])
    st.session_state[K_DEMO_SELECTION]     = payload.get("demo_selection")
    st.session_state[K_SUB_SELECTION]      = payload.get("sub_selection")
    if "code_to_text" in payload:
        st.session_state[K_CODE_TO_TEXT] = payload["code_to_text"]
    st.session_state[K_HAS_RESULTS] = True

def has_results() -> bool:
    return bool(st.session_state.get(K_HAS_RESULTS, False))

def get_results() -> Dict[str, Any]:
    """
    Returns a dict with the stashed results.
    Keys present (when has_results() is True):
      - tab_labels, pivot, per_q_disp, per_q_metric_col, per_q_metric_label,
        code_to_text, years, demo_selection, sub_selection
    """
    return {
        "tab_labels":         st.session_state.get(K_TAB_LABELS, []),
        "pivot":              st.session_state.get(K_PIVOT),
        "per_q_disp":         st.session_state.get(K_PER_Q_DISP, {}),
        "per_q_metric_col":   st.session_state.get(K_PER_Q_METRIC_COL, {}),
        "per_q_metric_label": st.session_state.get(K_PER_Q_METRIC_LABEL, {}),
        "code_to_text":       st.session_state.get(K_CODE_TO_TEXT, {}),
        "years":              st.session_state.get(K_SELECTED_YEARS, []),
        "demo_selection":     st.session_state.get(K_DEMO_SELECTION),
        "sub_selection":      st.session_state.get(K_SUB_SELECTION),
    }
