# menu1/render/results.py
from __future__ import annotations
from typing import Dict, Callable, Any, Tuple, List, Set, Optional
import io
import json
import hashlib
import re
import os

import pandas as pd
import streamlit as st

from ..ai import AI_SYSTEM_PROMPT  # unchanged

# ----------------------------- small helpers -----------------------------

def _hash_key(obj: Any) -> str:
    try:
        if isinstance(obj, pd.DataFrame):
            payload = obj.to_csv(index=True, na_rep="")
        else:
            payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        payload = str(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def _ai_cache_get(key: str):
    cache = st.session_state.get("menu1_ai_cache", {})
    return cache.get(key)

def _ai_cache_put(key: str, value: dict):
    cache = st.session_state.get("menu1_ai_cache", {})
    cache[key] = value
    st.session_state["menu1_ai_cache"] = cache

def _source_link_line(source_title: str, source_url: str) -> None:
    st.markdown(
        f"<div style='margin-top:6px; font-size:0.9rem;'>Source: "
        f"<a href='{source_url}' target='_blank'>{source_title}</a></div>",
        unsafe_allow_html=True
    )

def _find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df is None or df.empty:
        return None
    m = {c.lower(): c for c in df.columns}
    for n in names:
        c = m.get(n.lower())
        if c is not None:
            return c
    return None

def _has_data(df: pd.DataFrame, col: Optional[str]) -> bool:
    if not isinstance(df, pd.DataFrame) or df is None or df.empty:
        return False
    if not col or col not in df.columns:
        return False
    s = pd.to_numeric(df[col], errors="coerce")
    return s.notna().any()

def _safe_year_col(df: pd.DataFrame) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for c in ("Year", "year", "SURVEYR", "survey_year"):
        if c in df.columns:
            return c
    for c in df.columns:
        s = str(c)
        if len(s) == 4 and s.isdigit():
            return c
    return None

def _is_d57_exception(q: str) -> bool:
    qn = str(q).strip().upper().replace("-", "_")
    return qn in {"D57_A", "D57_B", "D57A", "D57B"}

def _sanitize_9999(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 9999 with NaN in numeric survey percentage columns, incl. Answer1..Answer6."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    # columns likely to be survey % values
    num_cols = []
    for c in out.columns:
        lc = str(c).lower().replace(" ", "")
        if lc in {"positive", "negative", "agree",
                  "answer1", "answer2", "answer3", "answer4", "answer5", "answer6"}:
            num_cols.append(c)
        elif re.fullmatch(r"answer\s*[1-6]", str(c), flags=re.IGNORECASE):
            num_cols.append(c)
    for c in set(num_cols):
        out[c] = pd.to_numeric(out[c], errors="coerce").replace(9999, pd.NA)
    return out

# ---------------------- metadata (polarity + scales) ----------------------

def _first_existing_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

@st.cache_data(show_spinner=False)
def _load_survey_questions_meta() -> pd.DataFrame:
    """
    Tries both metadata/ and project root for:
      - Survey Questions.xlsx
    Required: code, polarity (POS/NEG/NEU)
    Optional: positive/negative/agree (indices like "1,2")
              scale keys: scale / scale_id / scale_name
    """
    try:
        path = _first_existing_path([
            "metadata/Survey Questions.xlsx",
            "./Survey Questions.xlsx",
            "Survey Questions.xlsx",
        ])
        if not path:
            return pd.DataFrame(columns=["code", "polarity", "positive", "negative", "agree", "scale", "scale_id", "scale_name"])
        df = pd.read_excel(path)
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        if "question" in df.columns and "code" not in df.columns:
            df = df.rename(columns={"question": "code"})
        df["code"] = df["code"].astype(str).str.strip().str.upper()
        if "polarity" not in df.columns:
            df["polarity"] = "POS"
        df["polarity"] = df["polarity"].astype(str).str.upper().str.strip()
        for c in ("positive", "negative", "agree"):
            if c not in df.columns:
                df[c] = None
            else:
                df[c] = df[c].apply(lambda x: str(x).strip() if pd.notna(x) else None)
        if "scale" not in df.columns:
            df["scale"] = None
        if "scale_id" not in df.columns:
            df["scale_id"] = None
        if "scale_name" not in df.columns:
            df["scale_name"] = None
        return df
    except Exception:
        return pd.DataFrame(columns=["code", "polarity", "positive", "negative", "agree", "scale", "scale_id", "scale_name"])

@st.cache_data(show_spinner=False)
def _load_survey_scales_meta() -> pd.DataFrame:
    """
    Tries both metadata/ and project root for:
      - Survey Scales.xlsx
    Normalizes to: code (question code OR scale key), value (index), label (text)
    """
    try:
        path = _first_existing_path([
            "metadata/Survey Scales.xlsx",
            "./Survey Scales.xlsx",
            "Survey Scales.xlsx",
        ])
        if not path:
            return pd.DataFrame(columns=["code","value","label"])
        df = pd.read_excel(path)
        df.columns = [c.strip().lower() for c in df.columns]
        def pick(opts):  # helper
            for n in opts:
                if n in df.columns:
                    return n
            return None
        c_code  = pick(["code","question","qcode","qid","scale","scale_id","scale_key","scale_name","item"])
        c_val   = pick(["value","option","index","answer","answer_value","order","position"])
        c_label = pick(["label","answer_label","option_label","text","desc","description"])
        if not (c_code and c_val and c_label):
            return pd.DataFrame(columns=["code","value","label"])
        out = df[[c_code, c_val, c_label]].copy()
        out.columns = ["code", "value", "label"]
        out["code"] = out["code"].astype(str).str.strip().str.upper()
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["code","value","label"])
        out["value"] = out["value"].astype(int)
        out["label"] = out["label"].astype(str).str.strip()
        return out
    except Exception:
        return pd.DataFrame(columns=["code","value","label"])

def _parse_index_list(s: Optional[str]) -> List[int]:
    if not s or not isinstance(s, str):
        return []
    toks = re.split(r"[,\;\|\s]+", s.strip())
    out: List[int] = []
    for t in toks:
        if not t:
            continue
        try:
            out.append(int(float(t)))
        except Exception:
            continue
    return out

def _infer_reporting_field(metric_col: Optional[str]) -> Optional[str]:
    if not metric_col:
        return None
    lc = metric_col.replace(" ", "").lower()
    if lc == "positive": return "POSITIVE"
    if lc == "negative": return "NEGATIVE"
    if lc == "agree":    return "AGREE"
    if lc in ("answer1","answer_1"): return "ANSWER1"
    return None

# ----------- canonical scales (fallback) -----------

_CANONICAL_SCALES: Dict[str, List[str]] = {
    "AGREE5": [
        "Strongly disagree",
        "Disagree",
        "Neither agree nor disagree",
        "Agree",
        "Strongly agree",
    ],
    "EXTENT5": [
        "To a very small extent",
        "To a small extent",
        "To a moderate extent",
        "To a large extent",
        "To a very large extent",
    ],
}

def _pick_fallback_scale(question_text: Optional[str]) -> str:
    qt = (question_text or "").lower()
    if "to what extent" in qt or "extent" in qt:
        return "EXTENT5"
    return "AGREE5"

def _meaning_labels_for_question(
    *,
    qcode: str,
    question_text: Optional[str],
    reporting_field: Optional[str],
    metric_label: str,
    meta_q: pd.DataFrame,
    meta_scales: pd.DataFrame
) -> List[str]:
    """
    Return list of aggregated option labels (e.g., ["Strongly agree","Agree"]).
    Resolution order:
      1) Indices from Survey Questions row -> labels from Survey Scales where scales.code == question code.
      2) If none, look for a 'scale' key in Survey Questions row (scale/scale_id/scale_name) and map there.
      3) If still none:
         • choose canonical scale by question text (EXTENT5 for “extent”, else AGREE5)
         • slice by indices if present
         • SPECIAL PATCH: if no indices present AND reporting_field == NEGATIVE AND scale == EXTENT5,
           assume indices [2,3,4,5] (small→very large extent). This targets Q44x.
      4) As a last resort, derive from metric_label (“% selecting A / B”).
    """
    try:
        qU = str(qcode).strip().upper()

        # 1) indices from Survey Questions row
        idxs: List[int] = []
        row = meta_q[meta_q["code"] == qU]
        if reporting_field and not row.empty:
            colname = reporting_field.lower()  # positive/negative/agree/answer1
            if colname in row.columns:
                idxs = _parse_index_list(row.iloc[0][colname])

        labels: List[str] = []

        # helper to map indices using a given key into Survey Scales
        def _map_by_scales_key(key: Optional[str]) -> List[str]:
            if not key:
                return []
            sc = meta_scales[meta_scales["code"] == str(key).strip().upper()]
            if sc.empty:
                return []
            m = {int(v): str(l) for v, l in zip(sc["value"], sc["label"])}
            out: List[str] = []
            for i in idxs:
                if i in m:
                    out.append(m[i])
            return out

        # Try by question code
        if idxs:
            labels = _map_by_scales_key(qU)

        # Try by a scale key from Survey Questions (scale/scale_id/scale_name)
        if not labels and idxs and not row.empty:
            for sk in ("scale", "scale_id", "scale_name"):
                if sk in row.columns:
                    key = row.iloc[0][sk]
                    if pd.notna(key) and str(key).strip():
                        labels = _map_by_scales_key(str(key))
                        if labels:
                            break

        # Canonical fallback: choose scale by question text and slice
        if not labels:
            scale_key = _pick_fallback_scale(question_text)
            full = _CANONICAL_SCALES.get(scale_key, [])
            effective_idxs = list(idxs)  # copy
            # SPECIAL: Q44x-like “extent” negative aggregate with missing idxs -> [2,3,4,5]
            if (not effective_idxs) and reporting_field and reporting_field.upper() == "NEGATIVE" and scale_key == "EXTENT5":
                effective_idxs = [2,3,4,5]
            if effective_idxs and full:
                labels = [full[i - 1] for i in effective_idxs if 1 <= i <= len(full)]

        # Derive from metric_label if still nothing
        if not labels and isinstance(metric_label, str):
            m = re.search(r"%\s*selecting\s*(.+)$", metric_label.strip(), flags=re.IGNORECASE)
            if m:
                tail = m.group(1).strip()
                parts = [p.strip(" []") for p in re.split(r"\s*/\s*", tail) if p.strip()]
                labels = parts

        # dedupe preserving order
        seen = set(); out = []
        for x in labels:
            k = (x or "").lower()
            if k and k not in seen:
                out.append(x); seen.add(k)
        return out
    except Exception:
        return []

# ---------------------- Summary pivot (polarity-aware) ----------------------

def _pick_metric_for_summary(dfq: pd.DataFrame, qcode: str, meta: pd.DataFrame) -> Tuple[Optional[str], str]:
    pol = None
    if not meta.empty:
        row = meta[meta["code"] == str(qcode).strip().upper()]
        if not row.empty:
            pol = str(row.iloc[0]["polarity"] or "").upper().strip()
    pol = pol or "POS"

    col_pos = _find_col(dfq, ["Positive", "POSITIVE"])
    col_neg = _find_col(dfq, ["Negative", "NEGATIVE"])
    col_ag  = _find_col(dfq, ["AGREE"])
    col_a1  = _find_col(dfq, ["Answer1", "Answer 1", "ANSWER1"])

    def choose(seq: List[Tuple[Optional[str], str]]) -> Tuple[Optional[str], str]:
        for c, lbl in seq:
            if _has_data(dfq, c):
                return c, lbl
        return None, ""

    if pol == "NEG":
        return choose([(col_neg, "% negative"), (col_ag, "% agree"), (col_a1, "% selecting Answer1"), (col_pos, "% positive")])
    if pol == "NEU":
        return choose([(col_ag, "% agree"), (col_a1, "% selecting Answer1"), (col_pos, "% positive"), (col_neg, "% negative")])
    return choose([(col_pos, "% positive"), (col_ag, "% agree"), (col_a1, "% selecting Answer1"), (col_neg, "% negative")])

def _build_summary_pivot_from_disp(
    *,
    per_q_disp: Dict[str, pd.DataFrame],
    tab_labels: List[str],
    meta: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rows: List[Dict[str, Any]] = []
    labels_used: Dict[str, str] = {}

    for q in tab_labels:
        if _is_d57_exception(q):
            continue

        dfq_raw = per_q_disp.get(q)
        if not isinstance(dfq_raw, pd.DataFrame) or dfq_raw.empty:
            continue
        dfq = _sanitize_9999(dfq_raw)

        ycol = _safe_year_col(dfq)
        if not ycol:
            continue

        metric_col, metric_label = _pick_metric_for_summary(dfq, q, meta)
        if not metric_col:
            continue

        has_demo = "Demographic" in dfq.columns and dfq["Demographic"].astype(str).nunique(dropna=True) > 1
        group_cols = [ycol, metric_col] + (["Demographic"] if has_demo else [])
        tmp = dfq[group_cols].copy()

        tmp[ycol] = pd.to_numeric(tmp[ycol], errors="coerce")
        tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
        tmp = tmp.replace(9999, pd.NA)
        tmp = tmp.dropna(subset=[ycol, metric_col])

        if has_demo:
            tmp["Demographic"] = tmp["Demographic"].astype(str)
            for _, r in tmp.iterrows():
                rows.append({
                    "Question": q,
                    "Demographic": r["Demographic"],
                    "Year": int(r[ycol]),
                    "Value": float(r[metric_col]),
                })
        else:
            for _, r in tmp.iterrows():
                rows.append({
                    "Question": q,
                    "Demographic": "",
                    "Year": int(r[ycol]),
                    "Value": float(r[metric_col]),
                })

        labels_used[q] = metric_label or "% value"

    if not rows:
        return pd.DataFrame(), labels_used

    df = pd.DataFrame(rows)
    has_any_demo = df["Demographic"].astype(str).str.len().gt(0).any()
    index_cols = ["Question", "Demographic"] if has_any_demo else ["Question"]

    pivot = df.pivot_table(index=index_cols, columns="Year", values="Value", aggfunc="first").sort_index()
    pivot = pivot.applymap(lambda v: round(v, 1) if pd.notna(v) else v)
    return pivot, labels_used

# ---------------------- Validator helpers (per-question) ----------------------

def _is_year_like(n: int) -> bool:
    return 1900 <= n <= 2100

def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip().lower() in ("", "na", "n/a", "none", "nan", "null"):
            return None
        if x == 9999:
            return None
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return None
        return int(round(float(v)))
    except Exception:
        return None

def _pick_display_metric(df: pd.DataFrame, prefer: Optional[str] = None) -> Optional[str]:
    if prefer and prefer in df.columns:
        return prefer
    for c in ("value_display", "AGREE", "SCORE100"):
        if c in df.columns:
            return c
    return None

def _allowed_numbers_from_disp(df: pd.DataFrame, metric_col: str) -> Tuple[Set[int], Set[int]]:
    if df is None or df.empty:
        return set(), set()
    work = df.copy()
    if metric_col in work.columns:
        work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce").replace(9999, pd.NA)

    years_col = None
    for c in work.columns:
        if c.lower() == "year":
            years_col = c
            break
    if years_col is None:
        ycols = [c for c in work.columns if str(c).isdigit() and len(str(c)) == 4]
        if not ycols or metric_col not in work.columns:
            return set(), set()
        id_cols = [c for c in work.columns if c not in ycols]
        work = work.melt(id_vars=id_cols, value_vars=ycols, var_name="Year", value_name=metric_col)
        years_col = "Year"

    work["__Y__"] = pd.to_numeric(work[years_col], errors="coerce")
    work["__V__"] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=["__Y__", "__V__"])
    work["__Y__"] = work["__Y__"].astype(int).astype("Int64")
    work["__V__"] = work["__V__"].round().astype(int).astype("Int64")

    years = set([int(y) for y in work["__Y__"].dropna().unique().tolist() if _is_year_like(int(y))])
    allowed: Set[int] = set([int(v) for v in work["__V__"].dropna().unique().tolist()])

    if "Demographic" in work.columns:
        groups = list(work["Demographic"].astype(str).unique())
        for g in groups:
            sub = work[work["Demographic"].astype(str) == g].sort_values("__Y__")
            vals = sub["__V__"].dropna().astype(int).tolist()
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    allowed.add(abs(vals[j] - vals[i]))
    else:
        sub = work.sort_values("__Y__")
        vals = sub["__V__"].dropna().astype(int).tolist()
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                allowed.add(abs(vals[j] - vals[i]))

    if years:
        latest = max(years)
        latest_rows = work[work["__Y__"] == latest]
        if "Demographic" in latest_rows.columns:
            latest_rows = latest_rows.sort_values("Demographic")
        for i in range(len(latest_rows)):
            for j in range(i + 1, len(latest_rows)):
                try:
                    vi = int(latest_rows.iloc[i]["__V__"])
                    vj = int(latest_rows.iloc[j]["__V__"])
                    allowed.add(abs(vi - vj))
                except Exception:
                    pass
    return allowed, years

def _extract_datapoint_integers_with_sentences(text: str) -> List[Tuple[int, str]]:
    if not text:
        return []
    sentences = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    found: List[Tuple[int, str]] = []
    patterns = [
        re.compile(r"\((\d+)\)"),
        re.compile(r"\b(\d+)\s*points?\b", re.IGNORECASE),
        re.compile(r"\b(\d+)\s*%\b"),
        re.compile(r"\b(?:is|was|were|at|reached|stood(?:\s+at)?)\s+(\d+)\b", re.IGNORECASE),
        re.compile(r"\bfrom\s+(\d+)\b", re.IGNORECASE),
        re.compile(r"\bto\s+(\d+)\b", re.IGNORECASE),
        re.compile(r"\bvs\.?\s+(\d+)\b", re.IGNORECASE),
    ]
    for s in sentences:
        nums: Set[int] = set()
        for pat in patterns:
            for m in pat.finditer(s):
                try:
                    n = int(m.group(1))
                    nums.add(n)
                except Exception:
                    continue
        for n in sorted(nums):
            found.append((n, s))
    return found

def _validate_narrative(narrative: str, allowed: Set[int], years: Set[int]) -> dict:
    if not narrative:
        return {"ok": True, "bad_numbers": set(), "problems": []}
    pairs = _extract_datapoint_integers_with_sentences(narrative)
    bad: Set[int] = set()
    problems: List[str] = []
    for n, sentence in pairs:
        if _is_year_like(n):
            continue
        if n not in allowed:
            bad.add(n)
            problems.append(f"{n} — {sentence}")
    return {"ok": len(bad) == 0, "bad_numbers": bad, "problems": problems[:5]}

# ------------------- overall validator against summary_pivot ------------------

def _allowed_numbers_from_summary_pivot(pivot: pd.DataFrame) -> Tuple[Set[int], Set[int]]:
    if pivot is None or pivot.empty:
        return set(), set()
    work = pivot.copy()
    for c in work.columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    years = [int(c) for c in work.columns if str(c).isdigit() and len(str(c)) == 4]
    years_set = set(years)
    allowed: Set[int] = set()

    for v in work.values.flatten():
        if pd.notna(v):
            allowed.add(int(round(float(v))))
    for _, row in work.iterrows():
        vals = [row.get(y) for y in years]
        vals = [int(round(float(v))) for v in vals if pd.notna(v)]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                allowed.add(abs(vals[j] - vals[i]))
    for y in years:
        col = work[y]
        pairs = []
        for i in range(len(col)):
            for j in range(i + 1, len(col)):
                vi = col.iat[i]; vj = col.iat[j]
                if pd.notna(vi) and pd.notna(vj):
                    gap = abs(int(round(float(vi))) - int(round(float(vj))))
                    allowed.add(gap)
                    pairs.append(((i, j), gap))
        prev = [yy for yy in years if yy < y]
        if not prev:
            continue
        py = max(prev)
        col_prev = work[py]
        for (i, j), gap_latest in pairs:
            vi = col_prev.iat[i] if i < len(col_prev) else None
            vj = col_prev.iat[j] if j < len(col_prev) else None
            if pd.notna(vi) and pd.notna(vj):
                gap_prev = abs(int(round(float(vi))) - int(round(float(vj))))
                allowed.add(abs(gap_latest - gap_prev))
    return allowed, years_set

# ==================== AI narrative computation (unchanged core) ====================

def _compute_ai_narratives(
    *,
    tab_labels: List[str],
    per_q_disp: Dict[str, pd.DataFrame],
    per_q_metric_col: Dict[str, str],
    per_q_metric_label: Dict[str, str],
    code_to_text: Dict[str, str],
    demo_selection: Optional[str],
    pivot: pd.DataFrame,
    build_overall_prompt: Callable[..., str],
    build_per_q_prompt: Callable[..., str],
    call_openai_json: Callable[..., Tuple[Optional[str], Optional[str]]],
) -> Tuple[Dict[str, str], Optional[str]]:
    per_q_narratives: Dict[str, str] = {}
    overall_narrative: Optional[str] = None

    for q in tab_labels:
        df_disp = per_q_disp[q]
        metric_col = per_q_metric_col[q]
        metric_label = per_q_metric_label[q]
        qtext = code_to_text.get(q, "")
        content, _hint = call_openai_json(
            system=AI_SYSTEM_PROMPT,
            user=build_per_q_prompt(
                question_code=q,
                question_text=qtext,
                df_disp=df_disp,
                metric_col=metric_col,
                metric_label=metric_label,
                category_in_play=("Demographic" in df_disp.columns and df_disp["Demographic"].astype(str).nunique(dropna=True) > 1)
            )
        )
        try:
            j = json.loads(content) if content else {}
            per_q_narratives[q] = (j.get("narrative") or "").strip()
        except Exception:
            per_q_narratives[q] = ""

    if len(tab_labels) > 1:
        content, _hint = call_openai_json(
            system=AI_SYSTEM_PROMPT,
            user=build_overall_prompt(
                tab_labels=tab_labels,
                pivot_df=pivot,
                q_to_metric={q: per_q_metric_label[q] for q in tab_labels},
                code_to_text=code_to_text,
            )
        )
        try:
            j = json.loads(content) if content else {}
            overall_narrative = (j.get("narrative") or "").strip()
        except Exception:
            overall_narrative = None

    return per_q_narratives, overall_narrative

# ----- AI Data Validation (per-question; unchanged) --------------------------

def _render_data_validation_subsection(
    *,
    tab_labels: List[str],
    per_q_disp: Dict[str, pd.DataFrame],
    per_q_metric_col: Dict[str, str],
    per_q_narratives: Dict[str, str],
) -> None:
    any_issue = False
    details: List[Tuple[str, str]] = []

    for q in tab_labels:
        try:
            df_disp = per_q_disp.get(q)
            if not isinstance(df_disp, pd.DataFrame) or df_disp.empty:
                details.append(("caption", f"{q}: validation skipped (no table available)."))
                continue

            metric_col = per_q_metric_col.get(q) or _pick_display_metric(df_disp)
            if not metric_col:
                details.append(("caption", f"{q}: validation skipped (no metric column)."))
                continue

            df_val = df_disp.copy()
            if metric_col in df_val.columns:
                df_val[metric_col] = pd.to_numeric(df_val[metric_col], errors="coerce").replace(9999, pd.NA)

            allowed, years = _allowed_numbers_from_disp(df_val, metric_col)
            narrative = per_q_narratives.get(q, "") or ""
            res = _validate_narrative(narrative, allowed, years)

            if not res["ok"]:
                any_issue = True
                nums = ", ".join(str(x) for x in sorted(res["bad_numbers"]))
                details.append(("warning", f"{q}: potential mismatches detected ({nums})."))
            else:
                details.append(("caption", f"{q}: no numeric inconsistencies detected."))

        except Exception as e:
            details.append(("caption", f"{q}: validation skipped ({type(e).__name__})."))

    st.markdown(
        """
        <style>
          #ai_data_validation_title {
            font-size: 1rem;
            line-height: 1.25;
            font-weight: 600;
            margin: 0.25rem 0 0.25rem 0;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div id='ai_data_validation_title'>AI Data Validation</div>", unsafe_allow_html=True)
    st.markdown("✅ The data points in the summaries have been validated and correspond to the data provided." if not any_issue
                else "❌ Some AI statements may not match the tables. Review the details below.")
    with st.expander("View per-question validation details", expanded=False):
        for level, msg in details:
            st.warning(msg) if level == "warning" else st.caption(msg)

# ------------------------------ main renderer -------------------------------

def _meaning_labels_for_build(q: str, qtext: str, metric_col: Optional[str], metric_label: str,
                              meta_q: pd.DataFrame, meta_scales: pd.DataFrame) -> List[str]:
    """Helper to produce meaning_labels for the exact metric chosen for a question."""
    reporting_field = _infer_reporting_field(metric_col)
    return _meaning_labels_for_question(
        qcode=q,
        question_text=qtext,
        reporting_field=reporting_field,
        metric_label=metric_label or "",
        meta_q=meta_q,
        meta_scales=meta_scales
    )

# ----- NEW helpers for footnote presentation -----

_percent_pat_foot = re.compile(r"(\d{1,3})\s*%")

def _insert_first_percent_asterisk(text: str) -> str:
    """Insert a single asterisk immediately after the first percentage occurrence (e.g., '54%' or '54 %')."""
    if not text:
        return text
    m = _percent_pat_foot.search(text)
    if not m:
        return text
    i = m.end()
    if i < len(text) and text[i] == "*":
        return text
    return text[:i] + "*" + text[i:]

def _compress_labels_for_footnote(labels: List[str]) -> Optional[str]:
    """
    Return a compressed label string in parentheses, e.g.,
    ["To a small extent","To a moderate extent","To a large extent","To a very large extent"]
      -> "(To a small/moderate/large extent/very large extent)".
    Falls back to full join if safe compression not possible.
    """
    if not labels:
        return None
    full = "(" + "/".join(labels) + ")"
    try:
        from os.path import commonprefix
        prefix = commonprefix(labels)
        rev = [s[::-1] for s in labels]
        suffix = commonprefix(rev)[::-1]
        parts: List[str] = []
        for i, lab in enumerate(labels):
            core = lab
            if prefix and core.startswith(prefix):
                core = core[len(prefix):]
            if suffix and core.endswith(suffix):
                core = core[: -len(suffix)]
            if not core.strip():
                core = lab
            if i == 0 and lab.startswith(prefix):
                core = prefix + core
            if suffix and lab.endswith(suffix) and not core.endswith(suffix):
                core = core + suffix
            parts.append(core)
        compressed = "(" + "/".join(parts) + ")"
        if all(p.strip() for p in parts) and len(parts) == len(labels):
            return compressed
        return full
    except Exception:
        return full

def tabs_summary_and_per_q(
    *,
    payload: Dict[str, Any],
    ai_on: bool,
    build_overall_prompt: Callable[..., str],
    build_per_q_prompt: Callable[..., str],
    call_openai_json: Callable[..., Tuple[Optional[str], Optional[str]]],
    source_url: str,
    source_title: str,
) -> None:
    per_q_disp_in: Dict[str, pd.DataFrame] = payload["per_q_disp"]
    per_q_metric_col_in: Dict[str, str]   = payload["per_q_metric_col"]
    per_q_metric_label_in: Dict[str, str] = payload["per_q_metric_label"]
    pivot_from_payload: pd.DataFrame      = payload["pivot"]
    tab_labels                            = payload["tab_labels"]
    years                                 = payload["years"]
    demo_selection                        = payload["demo_selection"]
    sub_selection                         = payload["sub_selection"]
    code_to_text                          = payload["code_to_text"]

    # Sanitize per-question tables (9999 -> NaN)
    per_q_disp: Dict[str, pd.DataFrame] = {q: _sanitize_9999(df) for q, df in per_q_disp_in.items()}

    # Load metadata (now robust to root paths)
    meta_q = _load_survey_questions_meta()
    meta_scales = _load_survey_scales_meta()

    # Summary pivot (polarity-aware); exclude D57
    summary_pivot, labels_used = _build_summary_pivot_from_disp(
        per_q_disp=per_q_disp,
        tab_labels=tab_labels,
        meta=meta_q
    )

    # cache key
    ai_sig = {
        "tab_labels": tab_labels,
        "years": years,
        "demo_selection": demo_selection,
        "sub_selection": sub_selection,
        "metric_labels": {q: per_q_metric_label_in[q] for q in tab_labels},
        "pivot_sig": _hash_key(pivot_from_payload),
        "summary_sig": _hash_key(summary_pivot),
    }
    ai_key = "menu1_ai_" + _hash_key(ai_sig)

    # ------------------------ UX: header + tabs ------------------------
    st.header("Results")

    tab_titles = ["Summary table"] + tab_labels + ["Technical notes"]
    tabs = st.tabs(tab_titles)

    # Summary tab
    with tabs[0]:
        st.markdown("### Summary table")
        if tab_labels:
            st.markdown("<div style='font-size:0.9rem; color:#444; margin-bottom:4px;'>Questions & metrics included:</div>",
                        unsafe_allow_html=True)
            for q in tab_labels:
                if _is_d57_exception(q):
                    st.markdown(
                        f"<div style='font-size:0.85rem; color:#555;'><strong>{q}</strong>: {code_to_text.get(q, '')} "
                        f"<span style='opacity:.85;'>[distribution-only: Answer1..Answer6]</span></div>",
                        unsafe_allow_html=True
                    )
                    continue
                qtext = code_to_text.get(q, "")
                mlabel = labels_used.get(q) or per_q_metric_label_in.get(q, "% value")
                st.markdown(
                    f"<div style='font-size:0.85rem; color:#555;'><strong>{q}</strong>: {qtext} "
                    f"<span style='opacity:.85;'>[{mlabel}]</span></div>",
                    unsafe_allow_html=True
                )
        if summary_pivot is not None and not summary_pivot.empty:
            st.dataframe(summary_pivot.reset_index(), use_container_width=True)
        else:
            st.info("No data available for the summary under current filters.")
        _source_link_line(source_title, source_url)

    # Per-question tabs
    for idx, qcode in enumerate(tab_labels, start=1):
        with tabs[idx]:
            qtext = code_to_text.get(qcode, "")
            st.subheader(f"{qcode} — {qtext}")
            st.dataframe(per_q_disp[qcode], use_container_width=True)
            _source_link_line(source_title, source_url)

    # Technical notes
    tech_tab_index = len(tab_titles) - 1
    with tabs[tech_tab_index]:
        st.markdown("### Technical notes")
        st.markdown(
            """
1. **Summary results** are mainly shown as “positive answers,” reflecting the affirmative responses. Positive answers are calculated by removing the "Don't know" and "Not applicable" responses from the total responses.  
2. **Weights/adjustment:** Results have been adjusted for non-response to better represent the target population. Therefore, percentages should not be used to determine the number of respondents within a response category.  
3. **Rounding:** Due to rounding, percentages may not add to 100.  
4. **Suppression:** Results were suppressed for questions with low respondent counts (under 10) and for low response category counts.
            """
        )

    # ------------------------- AI section -------------------------
    if ai_on:
        st.markdown("---")
        st.markdown("## AI Summary")

        cached = _ai_cache_get(ai_key)
        if cached:
            per_q_narratives = cached.get("per_q", {}) or {}
            overall_narrative = cached.get("overall")

            # collect label strings to cite once under Overall
            overall_foot_labels: List[str] = []

            for q in tab_labels:
                txt = per_q_narratives.get(q, "")
                if txt:
                    st.markdown(f"**{q} — {code_to_text.get(q, '')}**")
                    # asterisk after first %
                    txt_star = _insert_first_percent_asterisk(txt)
                    st.write(txt_star)
                    # per-question footnote (skip D57)
                    try:
                        if not _is_d57_exception(q):
                            metric_col = ( _pick_metric_for_summary(per_q_disp[q], q, meta_q)[0]
                                           or per_q_metric_col_in.get(q) )
                            metric_label = labels_used.get(q) or per_q_metric_label_in.get(q, "% value")
                            qtext = code_to_text.get(q, "")
                            labels = _meaning_labels_for_question(
                                qcode=q,
                                question_text=qtext,
                                reporting_field=_infer_reporting_field(metric_col),
                                metric_label=metric_label or "",
                                meta_q=meta_q,
                                meta_scales=meta_scales
                            ) or []
                            if labels:
                                lab = _compress_labels_for_footnote(labels)
                                if lab:
                                    st.caption(f"* Percentages represent respondents’ aggregate answers: {lab}.")
                                    overall_foot_labels.append(f"{q} — {lab}")
                    except Exception:
                        pass

            if overall_narrative and len(tab_labels) > 1:
                st.markdown("**Overall**")
                st.write(overall_narrative)
                if overall_foot_labels:
                    st.caption(
                        "* In this section, percentages refer to the same aggregates used above: "
                        + "; ".join(overall_foot_labels) + "."
                    )
        else:
            # ---------- per-question AI ----------
            per_q_narratives: Dict[str, str] = {}
            q_to_meaning_labels: Dict[str, List[str]] = {}
            q_distribution_only: Dict[str, bool] = {}

            # collect label strings to cite once under Overall
            overall_foot_labels: List[str] = []

            for q in tab_labels:
                dfq = per_q_disp.get(q)
                qtext = code_to_text.get(q, "")

                if _is_d57_exception(q):
                    metric_col_ai = None
                    metric_label_ai = "Distribution (Answer1..Answer6)"
                    reporting_field_ai = None
                    category_in_play = ("Demographic" in dfq.columns and dfq["Demographic"].astype(str).nunique(dropna=True) > 1)
                    meaning_labels_ai: List[str] = []
                    q_distribution_only[q] = True
                else:
                    metric_col_ai, metric_label_ai = _pick_metric_for_summary(dfq, q, meta_q)
                    if not metric_col_ai:
                        metric_col_ai = per_q_metric_col_in.get(q)
                        metric_label_ai = per_q_metric_label_in.get(q, "% value")
                    else:
                        metric_label_ai = (metric_label_ai or labels_used.get(q) or per_q_metric_label_in.get(q, "% value"))
                    reporting_field_ai = _infer_reporting_field(metric_col_ai)
                    category_in_play = ("Demographic" in dfq.columns and dfq["Demographic"].astype(str).nunique(dropna=True) > 1)

                    # meaning labels with robust fallbacks + special NEGATIVE/EXTENT default
                    meaning_labels_ai = _meaning_labels_for_question(
                        qcode=q,
                        question_text=qtext,
                        reporting_field=reporting_field_ai,
                        metric_label=metric_label_ai or "",
                        meta_q=meta_q,
                        meta_scales=meta_scales
                    )
                    q_distribution_only[q] = False

                q_to_meaning_labels[q] = meaning_labels_ai

                with st.spinner(f"AI — analyzing {q}…"):
                    try:
                        content, _hint = call_openai_json(
                            system=AI_SYSTEM_PROMPT,
                            user=build_per_q_prompt(
                                question_code=q,
                                question_text=qtext,
                                df_disp=(dfq.copy(deep=True) if isinstance(dfq, pd.DataFrame) else dfq),
                                metric_col=metric_col_ai,
                                metric_label=metric_label_ai,
                                category_in_play=category_in_play,
                                meaning_labels=meaning_labels_ai,
                                reporting_field=reporting_field_ai,
                                distribution_only=_is_d57_exception(q)
                            )
                        )
                        j = json.loads(content) if content else {}
                        txt = (j.get("narrative") or "").strip()
                    except Exception as e:
                        txt = ""
                        st.warning(f"AI skipped for {q} due to an internal error ({type(e).__name__}).")
                per_q_narratives[q] = txt
                if txt:
                    st.markdown(f"**{q} — {qtext}**")
                    txt_star = _insert_first_percent_asterisk(txt)
                    st.write(txt_star)
                    if meaning_labels_ai:
                        lab = _compress_labels_for_footnote(meaning_labels_ai)
                        if lab:
                            st.caption(f"* Percentages represent respondents’ aggregate answers: {lab}.")
                            overall_foot_labels.append(f"{q} — {lab}")

            # ---------- OVERALL ----------
            overall_narrative = None
            if len(tab_labels) > 1 and summary_pivot is not None and not summary_pivot.empty:
                with st.spinner("AI — synthesizing overall pattern…"):
                    try:
                        content, _hint = call_openai_json(
                            system=AI_SYSTEM_PROMPT,
                            user=build_overall_prompt(
                                tab_labels=tab_labels,
                                pivot_df=summary_pivot.copy(deep=True),
                                q_to_metric={q: (labels_used.get(q) or per_q_metric_label_in[q]) for q in tab_labels},
                                code_to_text=code_to_text,
                                q_to_meaning_labels=q_to_meaning_labels,
                                q_distribution_only=q_distribution_only
                            )
                        )
                    except Exception as e:
                        content = None
                        st.warning(f"AI skipped for Overall due to an internal error ({type(e).__name__}).")
                    try:
                        j = json.loads(content) if content else {}
                        overall_narrative = (j.get("narrative") or "").strip()
                    except Exception:
                        overall_narrative = None

                if overall_narrative:
                    st.markdown("**Overall**")
                    st.write(overall_narrative)
                    if overall_foot_labels:
                        st.caption(
                            "* In this section, percentages refer to the same aggregates used above: "
                            + "; ".join(overall_foot_labels) + "."
                        )

            _ai_cache_put(ai_key, {"per_q": per_q_narratives, "overall": overall_narrative})
            st.session_state["menu1_ai_narr_per_q"] = per_q_narratives
            st.session_state["menu1_ai_narr_overall"] = overall_narrative

        # ---------- AI Data Validation ----------
        try:
            _render_data_validation_subsection(
                tab_labels=tab_labels,
                per_q_disp=per_q_disp,
                per_q_metric_col={
                    q: (None if _is_d57_exception(q) else (_pick_metric_for_summary(per_q_disp[q], q, meta_q)[0] or per_q_metric_col_in.get(q)))
                    for q in tab_labels
                },
                per_q_narratives=st.session_state.get("menu1_ai_narr_per_q", {}) or {},
            )
            if len(tab_labels) > 1 and summary_pivot is not None and not summary_pivot.empty:
                overall_txt = st.session_state.get("menu1_ai_narr_overall", "")
                if overall_txt:
                    allowed, years_set = _allowed_numbers_from_summary_pivot(summary_pivot.copy(deep=True))
                    res = _validate_narrative(overall_txt, allowed, years_set)
                    if not res["ok"]:
                        st.warning("Overall: potential mismatches detected in the AI summary.")
                        with st.expander("View overall validation details"):
                            for prob in res["problems"]:
                                st.caption(prob)
                    else:
                        st.caption("Overall: no numeric inconsistencies detected.")
        except Exception:
            st.caption("AI Data Validation is unavailable for this run.")

    # ----------------------- Footer: Export + Start new -----------------------
    st.markdown("---")
    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
    col_dl, col_new = st.columns([1, 1])

    with col_dl:
        export_per_q = st.session_state.get("menu1_ai_narr_per_q")
        export_overall = st.session_state.get("menu1_ai_narr_overall")

        if ai_on and export_per_q is None:
            cached = _ai_cache_get(ai_key)
            if cached:
                export_per_q = cached.get("per_q", {})
                export_overall = cached.get("overall")
            else:
                try:
                    per_q_disp_ai2: Dict[str, pd.DataFrame] = {}
                    for q in tab_labels:
                        dfq = per_q_disp.get(q)
                        per_q_disp_ai2[q] = (dfq.copy(deep=True) if isinstance(dfq, pd.DataFrame) else dfq)
                    export_per_q, export_overall = _compute_ai_narratives(
                        tab_labels=tab_labels,
                        per_q_disp=per_q_disp_ai2,
                        per_q_metric_col={
                            q: (_pick_metric_for_summary(per_q_disp[q], q, meta_q)[0] or per_q_metric_col_in.get(q))
                            for q in tab_labels
                        },
                        per_q_metric_label={q: (labels_used.get(q) or per_q_metric_label_in.get(q, "% value")) for q in tab_labels},
                        code_to_text=code_to_text,
                        demo_selection=demo_selection,
                        pivot=summary_pivot.copy(deep=True) if (summary_pivot is not None and not summary_pivot.empty) else pivot_from_payload.copy(deep=True),
                        build_overall_prompt=build_overall_prompt,
                        build_per_q_prompt=build_per_q_prompt,
                        call_openai_json=call_openai_json,
                    )
                    _ai_cache_put(ai_key, {"per_q": export_per_q, "overall": export_overall})
                except Exception:
                    export_per_q, export_overall = {}, None

        _render_excel_download(
            summary_pivot=summary_pivot,
            per_q_disp=per_q_disp,
            tab_labels=tab_labels,
            per_q_narratives=(export_per_q or {}),
            overall_narrative=(export_overall if (export_overall and len(tab_labels) > 1) else None),
        )

    with col_new:
        if st.button("Start a new search", key="menu1_new_search"):
            _prev_ai_toggle = st.session_state.get("menu1_ai_toggle")
            try:
                from .. import state
                state.reset_menu1_state()
            except Exception:
                for k in [
                    "menu1_selected_codes", "menu1_multi_questions",
                    "menu1_show_diag",
                    "select_all_years", "demo_main", "last_query_info",
                ]:
                    st.session_state.pop(k, None)
            for k in [
                "menu1_hits", "menu1_hit_codes_selected",
                "menu1_search_done", "menu1_last_search_query",
                "menu1_kw_query",
            ]:
                st.session_state.pop(k, None)
            for k in list(st.session_state.keys()):
                if k.startswith("kwhit_") or k.startswith("sel_"):
                    st.session_state.pop(k, None)
            st.session_state.pop("menu1_global_hits_selected", None)
            st.session_state.pop("menu1_ai_cache", None)
            st.session_state.pop("menu1_ai_narr_per_q", None)
            st.session_state.pop("menu1_ai_narr_overall", None)
            if _prev_ai_toggle is not None:
                st.session_state["menu1_ai_toggle"] = _prev_ai_toggle
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

# ------------------- Excel export --------------------

def _render_excel_download(
    *,
    summary_pivot: pd.DataFrame,
    per_q_disp: Dict[str, pd.DataFrame],
    tab_labels: List[str],
    per_q_narratives: Dict[str, str],
    overall_narrative: Optional[str],
) -> None:
    with io.BytesIO() as buf:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            if summary_pivot is not None and not summary_pivot.empty:
                summary_pivot.reset_index().to_excel(writer, sheet_name="Summary_Table", index=False)
            else:
                pd.DataFrame({"Message": ["No data available for the summary under current filters."]}).to_excel(
                    writer, sheet_name="Summary_Table", index=False
                )
            for q, df_disp in per_q_disp.items():
                safe = q[:28]
                df_disp.to_excel(writer, sheet_name=f"{safe}", index=False)
            rows = []
            for q in tab_labels:
                txt = per_q_narratives.get(q, "")
                if txt:
                    rows.append({"Section": q, "Narrative": txt})
            if overall_narrative and len(tab_labels) > 1:
                rows.append({"Section": "Overall", "Narrative": overall_narrative})
            ai_df = pd.DataFrame(rows, columns=["Section", "Narrative"])
            ai_df.to_excel(writer, sheet_name="AI Summary", index=False)
        data = buf.getvalue()

    st.download_button(
        label="Download data and AI summaries",
        data=data,
        file_name="PSES_results_with_AI.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="menu1_excel_download",
    )
