# Add near your other config constants at top-level:
PSES_FORCE_INMEM_PREWARM = os.environ.get("PSES_FORCE_INMEM_PREWARM", "0").strip().lower() in ("1", "true", "yes")


@st.cache_resource(show_spinner="⚡ Preloading metadata & backend…")
def prewarm_all() -> dict:
    """
    SAFE prewarm:
      - Ensure CSV exists
      - Try to ensure Parquet dataset (fast query path)
      - Load metadata
      - Try to preload PS-wide df into memory, but NEVER fail the app if it’s too heavy
    """
    # 1) Ensure raw results are present (download)
    csv_path = ensure_results2024_local()

    # 2) Prefer Parquet if available (build if needed) — but don’t fail app if it can’t build
    engine = "csv"
    parquet_err = None
    try:
        if _pyarrow_available():
            ensure_parquet_dataset()
            engine = "parquet"
    except Exception as e:
        parquet_err = f"{type(e).__name__}: {e}"
        engine = "csv"

    # 3) Metadata (cached)
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

    # 4) In-memory PS-wide DataFrame
    # IMPORTANT: This is the most likely OOM point.
    inmem_ok = False
    inmem_err = None
    inmem_rows = 0

    if PSES_FORCE_INMEM_PREWARM:
        try:
            df_inmem = preload_pswide_dataframe()
            inmem_rows = int(df_inmem.shape[0]) if isinstance(df_inmem, pd.DataFrame) else 0
            inmem_ok = inmem_rows > 0
        except Exception as e:
            # DO NOT crash the app during prewarm
            inmem_err = f"{type(e).__name__}: {e}"
            inmem_ok = False
            inmem_rows = 0
    else:
        # Skip in-memory preload by default to avoid Streamlit OOM kills
        inmem_ok = False
        inmem_rows = 0

    _INMEM_STATE.update({
        "mode": f"pswide_df({engine})" if inmem_ok else "skipped",
        "rows": inmem_rows,
    })

    global _LAST_ENGINE
    _LAST_ENGINE = engine

    return {
        "engine": engine,
        "parquet_error": parquet_err,
        "inmem_mode": _INMEM_STATE["mode"],
        "inmem_rows": _INMEM_STATE["rows"],
        "inmem_ok": inmem_ok,
        "inmem_error": inmem_err,
        "metadata_counts": dict(_META_CACHE["counts"]),
        "pswide_only": True,
        "parquet_dir": PARQUET_ROOTDIR,
        "csv_path": csv_path,
        "agree_src": _AGREE_SRC,
    }


@st.cache_resource(show_spinner="⚡ Warming up data backend…")
def prewarm_fastpath() -> str:
    summary = prewarm_all()
    return "parquet" if summary.get("engine") == "parquet" else "csv"


def get_backend_info() -> dict:
    try:
        summary = prewarm_all()
    except Exception:
        summary = {}
    return {
        "last_engine": summary.get("engine", _LAST_ENGINE),
        "parquet_error": summary.get("parquet_error"),
        "inmem_mode": summary.get("inmem_mode", _INMEM_STATE["mode"]),
        "inmem_rows": int(summary.get("inmem_rows", _INMEM_STATE["rows"] or 0)),
        "inmem_ok": bool(summary.get("inmem_ok", False)),
        "inmem_error": summary.get("inmem_error"),
        "metadata_counts": summary.get("metadata_counts", _META_CACHE.get("counts", {})),
        "pswide_only": summary.get("pswide_only", True),
        "parquet_dir": PARQUET_ROOTDIR,
        "csv_path": summary.get("csv_path", LOCAL_GZ_PATH),
        "agree_src": summary.get("agree_src", _AGREE_SRC),
    }
