# utils/hybrid_search.py
# -------------------------------------------------------------------------
# Lexical-first search with semantic complements (term-agnostic)
# Returns ALL lexical hits, and ALSO semantic hits for items without lexical
# evidence (UI separates by 'origin' = 'lex' or 'sem').
# Adds lightweight diagnostics helpers:
#   - get_embedding_status()
#   - get_last_search_metrics()
# -------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
import hashlib
import os
import re
import time
import pandas as pd

# Optional semantic support (graceful degradation)
_ST_OK: bool = False
_ST_MODEL = None
_ST_NAME = os.environ.get("MENU1_EMBED_MODEL", os.environ.get("PSES_EMBED_MODEL", "all-MiniLM-L6-v2"))

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np  # type: ignore
    _ST_OK = True
except Exception:
    _ST_OK = False

try:
    import torch  # type: ignore
    _TORCH_VER = getattr(torch, "__version__", None)
    _TORCH_DEV = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
except Exception:
    _TORCH_VER = None
    _TORCH_DEV = None

# -----------------------------
# Normalization / tokenization
# -----------------------------
_word_re = re.compile(r"[a-z0-9']+")
_stop = {
    "the","and","of","to","in","for","with","on","at","by","from",
    "a","an","is","are","was","were","be","been","being","or","as",
    "it","that","this","these","those","i","you","we","they","he","she"
}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def _tokens(s: str) -> List[str]:
    return [t for t in _word_re.findall(_norm(s)) if t and t not in _stop]

def _uniq(seq: List[str]) -> List[str]:
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

# -----------------------------
# Lightweight stemming + 4-grams
# -----------------------------
def _stem(tok: str) -> str:
    for suf in ("ments","ment","ings","ing","ities","ity","ions","ion",
                "ness","ships","ship","ably","able","ally","al","ed","es","s","y"):
        if tok.endswith(suf) and len(tok) > len(suf) + 2:
            return tok[: -len(suf)]
    return tok

def _stems(tokens: List[str]) -> List[str]:
    return [_stem(t) for t in tokens]

def _char4(tok: str) -> List[str]:
    return [tok[i:i+4] for i in range(len(tok)-3)] if len(tok) >= 4 else [tok]

# -----------------------------
# +include / -exclude parsing
# -----------------------------
def _parse_req_exc(raw_q: str) -> Tuple[str, List[str], List[str]]:
    parts = raw_q.split()
    inc, exc, kept = [], [], []
    for p in parts:
        if p.startswith("+") and len(p) > 1: inc.append(_norm(p[1:]))
        elif p.startswith("-") and len(p) > 1: exc.append(_norm(p[1:]))
        else: kept.append(p)
    return " ".join(kept).strip(), inc, exc

# -----------------------------
# Code-aware normalization / matching
# -----------------------------
_CODE_HINT_RE = re.compile(r"(?i)\b(?:q|question)\s*0*([0-9]+)\s*([a-z]?)\b")

def _split_code_parts(code: str) -> Tuple[Optional[int], str]:
    if not code:
        return None, ""
    s = _norm(code).replace(" ", "")
    m = re.match(r"^([a-z]+)?0*([0-9]+)([a-z]*)$", s)
    if not m:
        return None, ""
    _, num, suffix = m.groups()
    try:
        n = int(num)
    except Exception:
        return None, ""
    return n, (suffix or "")

def _extract_code_hints(raw_query: str) -> List[Tuple[int, str]]:
    hints: List[Tuple[int, str]] = []
    for num, suf in _CODE_HINT_RE.findall(raw_query or ""):
        try:
            hints.append((int(num), _norm(suf)))
        except Exception:
            pass
    collapsed = _norm((raw_query or "")).replace(" ", "")
    m = re.match(r"(?i)^(?:q|question)0*([0-9]+)([a-z]?)$", collapsed)
    if m:
        num, suf = m.groups()
        try:
            hints.append((int(num), _norm(suf)))
        except Exception:
            pass
    seen = set(); out = []
    for h in hints:
        if h not in seen:
            seen.add(h); out.append(h)
    return out

def _code_hint_matches_item(hint: Tuple[int, str], item_code: str) -> bool:
    n_item, suf_item = _split_code_parts(item_code)
    if n_item is None:
        return False
    n_hint, suf_hint = hint
    if n_hint != n_item:
        return False
    if not suf_hint:
        return True
    return suf_item.startswith(suf_hint)

# -----------------------------
# Embedding cache / status
# -----------------------------
_EMBED_CACHE: Dict[str, "np.ndarray"] = {}
_TXT_CACHE: Dict[str, List[str]] = {}
_LAST_SEARCH_METRICS: Dict[str, object] = {}  # exposed via get_last_search_metrics()

def _index_key(texts: List[str]) -> str:
    h = hashlib.md5()
    for t in texts:
        h.update((_norm(t)+"\n").encode("utf-8"))
    return h.hexdigest()

def _get_semantic_matrix(texts: List[str]) -> Optional["np.ndarray"]:
    if not _ST_OK: return None
    global _ST_MODEL
    if _ST_MODEL is None:
        try:
            _ST_MODEL = SentenceTransformer(_ST_NAME)
        except Exception:
            return None
    key = _index_key(texts)
    if key in _EMBED_CACHE and _TXT_CACHE.get(key) == texts:
        return _EMBED_CACHE[key]
    try:
        mat = _ST_MODEL.encode(texts, normalize_embeddings=True)
    except Exception:
        return None
    _EMBED_CACHE[key] = mat
    _TXT_CACHE[key] = texts
    return mat

def _cosine_sim(vecA, matB):  # both normalized
    return (matB @ vecA)

# -----------------------------
# IDF for 4-grams
# -----------------------------
_GRAM_DF: Dict[str, int] = {}
_GRAM_INFORMATIVE: Set[str] = set()
_GRAM_READY: bool = False

def _build_gram_df(texts: List[str]) -> None:
    global _GRAM_DF, _GRAM_INFORMATIVE, _GRAM_READY
    if _GRAM_READY:
        return
    df: Dict[str, int] = {}
    for txt in texts:
        grams = set()
        toks = _stems(_tokens(txt))
        for t in toks:
            grams.update(_char4(t))
        for g in grams:
            df[g] = df.get(g, 0) + 1
    if not df:
        _GRAM_DF = {}
        _GRAM_INFORMATIVE = set()
        _GRAM_READY = True
        return
    counts = sorted(df.values(), reverse=True)
    cutoff_index = int(0.15 * (len(counts)-1)) if len(counts) > 1 else 0
    cutoff_val = counts[cutoff_index] if counts else 0
    informative = {g for g, c in df.items() if c < cutoff_val}
    _GRAM_DF = df
    _GRAM_INFORMATIVE = informative
    _GRAM_READY = True

def _jaccard_informative_grams(qgrams: Set[str], tgrams: Set[str]) -> float:
    if not _GRAM_READY:
        return 0.0
    iq = qgrams & _GRAM_INFORMATIVE
    it = tgrams & _GRAM_INFORMATIVE
    if not iq and not it:
        return 0.0
    inter = len(iq & it)
    union = len(iq | it)
    return inter / union if union else 0.0

# -----------------------------
# Public entry point
# -----------------------------
def hybrid_question_search(
    qdf: pd.DataFrame,
    query: str,
    *,
    top_k: int = 120,
    min_score: float = 0.40,
) -> pd.DataFrame:
    """
    Returns DataFrame[code, text, display, score, origin] per policy.
    """
    t0 = time.time()
    t_lex0 = time.time()

    if qdf is None or qdf.empty or not query or not str(query).strip():
        return qdf.head(0)

    for col in ("code","text","display"):
        if col not in qdf.columns:
            raise ValueError(f"qdf missing required column: {col}")

    codes  = qdf["code"].astype(str).tolist()
    texts  = qdf["text"].astype(str).tolist()
    shows  = qdf["display"].astype(str).tolist()

    q_raw = str(query).strip()
    q_clean, includes, excludes = _parse_req_exc(q_raw)

    _build_gram_df(texts)

    qtoks  = _uniq(_tokens(q_clean))
    qstems = _stems(qtoks)
    qstem_set = set(qstems)
    qgrams = set(g for t in qstems for g in _char4(t))

    # Include CODE + DISPLAY + TEXT in haystack
    haystacks = [_norm(f"{code} {disp} {txt}") for code, disp, txt in zip(codes, shows, texts)]
    code_hints = _extract_code_hints(q_raw)

    N = len(texts)
    lex_scores = [0.0] * N
    has_lex    = [False] * N

    # --- Lexical evidence ---
    JACCARD_CUTOFF = 0.22  # relaxed (typo tolerant)

    for i, (code, txt, disp) in enumerate(zip(codes, texts, shows)):
        # Code-aware (strong)
        if any(_code_hint_matches_item(h, code) for h in code_hints):
            has_lex[i] = True
            lex_scores[i] = 1.0
            continue

        # Stems across TEXT + DISPLAY
        toks_text = _stems(_tokens(txt))
        toks_disp = _stems(_tokens(disp))
        toks_all  = set(toks_text) | set(toks_disp)
        stem_o    = len(qstem_set & toks_all)
        stem_cov  = (stem_o / max(1, len(qstems))) if qstems else 0.0

        # Informative 4-gram Jaccard
        grams_i = set(g for t in toks_all for g in _char4(t))
        jacc = _jaccard_informative_grams(qgrams, grams_i)

        is_lex = (stem_o > 0) or (jacc >= JACCARD_CUTOFF)
        if is_lex:
            stem_cov = max(stem_cov, 0.50)
            has_lex[i] = True
        lex_scores[i] = min(1.0, max(0.0, stem_cov))

    # include/exclude
    def _contains_any(hay: str, needles: List[str]) -> bool:
        return any(n and n in hay for n in needles)

    for i, hay in enumerate(haystacks):
        if excludes and _contains_any(hay, excludes):
            lex_scores[i] = 0.0; has_lex[i] = False
        if includes and not all(inc in hay for inc in includes):
            lex_scores[i] = 0.0; has_lex[i] = False

    df_all = pd.DataFrame({
        "code": codes, "text": texts, "display": shows, "score": lex_scores, "has_lex": has_lex
    })

    df_lex = df_all[(df_all["has_lex"]) & (df_all["score"] > float(min_score))].copy()
    df_lex["origin"] = "lex"
    df_lex = df_lex.sort_values(["score","code"], ascending=[False, True]).drop_duplicates("code", keep="first")

    t_lex1 = time.time()
    t_sem0 = time.time()

    # --- Semantic for NON-lex items ---
    df_nonlex = df_all[~df_all["has_lex"]].reset_index(drop=True)

    if _ST_OK and not df_nonlex.empty:
        try:
            mat = _get_semantic_matrix(texts)
            if mat is not None:
                global _ST_MODEL
                qvec = _ST_MODEL.encode([q_raw], normalize_embeddings=True)[0]
                sim  = _cosine_sim(qvec, mat)            # [-1,1]
                sem_all = ((sim + 1.0) / 2.0).tolist()    # [0,1]
            else:
                sem_all = [0.0]*N
        except Exception:
            sem_all = [0.0]*N
    else:
        sem_all = [0.0]*N

    SEM_FLOOR = 0.65  # lowered; pure-cosine (no anchor)

    sem_rows = []
    for i in range(N):
        if has_lex[i]:
            continue
        s = sem_all[i]
        if s < SEM_FLOOR:
            continue
        s_shaped = min(1.0, max(0.0, s * s))
        sem_rows.append((codes[i], texts[i], shows[i], s_shaped, "sem"))

    df_sem = pd.DataFrame(sem_rows, columns=["code","text","display","score","origin"]) \
             if sem_rows else pd.DataFrame(columns=["code","text","display","score","origin"])

    out = pd.concat([
        df_lex[["code","text","display","score","origin"]],
        df_sem[["code","text","display","score","origin"]],
    ], ignore_index=True)

    if out.empty:
        # record diagnostics even on empty
        _LAST_SEARCH_METRICS.update({
            "query": q_raw, "count_lex": int(len(df_lex)), "count_sem": int(len(df_sem)),
            "total": 0, "top_k": int(top_k), "min_score": float(min_score),
            "sem_floor": float(SEM_FLOOR), "jaccard_cutoff": float(JACCARD_CUTOFF),
            "semantic_active": bool(_ST_OK and (_ST_MODEL is not None)),
            "t_lex_ms": int((t_lex1 - t_lex0) * 1000),
            "t_sem_ms": int((time.time() - t_sem0) * 1000),
            "t_total_ms": int((time.time() - t0) * 1000),
        })
        return out

    out = out.sort_values("score", ascending=False).drop_duplicates("code", keep="first")
    out = out[out["score"] > float(min_score)]
    if top_k and top_k > 0:
        out = out.head(top_k)
    out = out.sort_values(["score","code"], ascending=[False, True]).reset_index(drop=True)

    t1 = time.time()
    _LAST_SEARCH_METRICS.update({
        "query": q_raw, "count_lex": int(len(df_lex)), "count_sem": int(len(df_sem)),
        "total": int(len(out)), "top_k": int(top_k), "min_score": float(min_score),
        "sem_floor": float(SEM_FLOOR), "jaccard_cutoff": float(JACCARD_CUTOFF),
        "semantic_active": bool(_ST_OK and (_ST_MODEL is not None)),
        "t_lex_ms": int((t_lex1 - t_lex0) * 1000),
        "t_sem_ms": int((t1 - t_sem0) * 1000),
        "t_total_ms": int((t1 - t0) * 1000),
    })
    return out


# -------------------------------------------------------------------------
# Diagnostics helpers (imported by Diagnostics panel)
# -------------------------------------------------------------------------
def get_embedding_status() -> Dict[str, object]:
    """Return a snapshot of semantic engine status (safe to call any time)."""
    try:
        import sentence_transformers as _st  # type: ignore
        st_ver = getattr(_st, "__version__", None)
    except Exception:
        st_ver = None
    status = {
        "semantic_library_installed": bool(_ST_OK),
        "sentence_transformers_version": st_ver,
        "torch_version": _TORCH_VER,
        "device": _TORCH_DEV or "cpu",
        "model_name": _ST_NAME,
        "model_loaded": bool(_ST_MODEL is not None),
        "embedding_index_ready": bool(_EMBED_CACHE),  # True after first build
        "catalogues_indexed": len(_EMBED_CACHE) or 0,
    }
    return status

def get_last_search_metrics() -> Dict[str, object]:
    """Return metrics of the last hybrid_question_search() call."""
    return dict(_LAST_SEARCH_METRICS)
