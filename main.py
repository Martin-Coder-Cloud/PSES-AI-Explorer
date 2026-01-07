# main.py — Home-first router; Menu 1 renders first, then loads backend with centered overlay spinner
from __future__ import annotations

import importlib
import time
import streamlit as st
import streamlit.components.v1 as components


# --- Quick exit for GitHub Action keepalive pings (avoids heavy loading) ---
def _is_keepalive_ping() -> bool:
    try:
        params = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()  # type: ignore[attr-defined]
        raw = params.get("keepalive", ["0"])
        val = raw[0] if isinstance(raw, list) else raw
        return str(val).lower() in ("1", "true", "yes")
    except Exception:
        return False


if _is_keepalive_ping():
    st.write("✅ App is awake.")
    st.stop()


# ── Make set_page_config idempotent ──────────────────────────────────────────
if not hasattr(st, "_setpcf_wrapped"):
    _orig_spc = st.set_page_config

    def _safe_set_page_config(*args, **kwargs):
        if st.session_state.get("_page_config_done"):
            return
        st.session_state["_page_config_done"] = True
        return _orig_spc(*args, **kwargs)

    st.set_page_config = _safe_set_page_config
    st._setpcf_wrapped = True

st.set_page_config(page_title="PSES Explorer", layout="wide")


# ── Load data loader (optional) ──────────────────────────────────────────────
_loader_err = ""
_dl = None
try:
    _dl = importlib.import_module("utils.data_loader")
except Exception as e:
    _loader_err = f"{type(e).__name__}: {e}"


def _fn(name, default=None):
    return getattr(_dl, name, default) if _dl else default


prewarm_all = _fn("prewarm_all")
preload_pswide_dataframe = _fn("preload_pswide_dataframe")  # may or may not exist


# ── Navigation helper ────────────────────────────────────────────────────────
def goto(page: str):
    st.session_state["_nav"] = page
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


# ── Remove hero background for non-Home pages ────────────────────────────────
def _clear_bg_css():
    st.markdown(
        """
        <style>
            .block-container {
                background-image: none !important;
                background: none !important;
                color: inherit !important;
                padding-top: 1.25rem !important;
                padding-left: 1.25rem !important;
                padding-bottom: 2rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Full-screen centered overlay spinner (HTML/CSS) ──────────────────────────
def _show_center_overlay(message: str = "Loading database into memory…"):
    # Fixed overlay so it is always centered and never “cut” by Streamlit header.
    components.html(
        f"""
        <style>
          .pses-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.35);
            z-index: 999999;
            display: flex;
            align-items: center;
            justify-content: center;
          }}
          .pses-overlay-card {{
            background: rgba(255,255,255,0.92);
            border-radius: 18px;
            padding: 22px 26px;
            box-shadow: 0 16px 50px rgba(0,0,0,0.35);
            min-width: 340px;
            max-width: 520px;
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
            color: #111827;
          }}
          .pses-row {{
            display: flex;
            gap: 14px;
            align-items: center;
          }}
          .pses-spinner {{
            width: 22px;
            height: 22px;
            border: 3px solid rgba(17,24,39,0.25);
            border-top-color: rgba(17,24,39,0.95);
            border-radius: 50%;
            animation: pses-spin 0.9s linear infinite;
            flex: 0 0 auto;
          }}
          @keyframes pses-spin {{
            to {{ transform: rotate(360deg); }}
          }}
          .pses-title {{
            font-size: 16px;
            font-weight: 700;
            margin: 0;
            line-height: 1.25;
          }}
          .pses-sub {{
            font-size: 13px;
            margin: 6px 0 0 0;
            opacity: 0.9;
            line-height: 1.35;
          }}
        </style>

        <div class="pses-overlay">
          <div class="pses-overlay-card">
            <div class="pses-row">
              <div class="pses-spinner"></div>
              <div>
                <p class="pses-title">{message}</p>
                <p class="pses-sub">One-time initialization. Queries will be much faster after this step.</p>
              </div>
            </div>
          </div>
        </div>
        """,
        height=0,
        width=0,
    )


# ── Home ─────────────────────────────────────────────────────────────────────
def render_home():
    if st.session_state.pop("_scroll_top_home", False):
        components.html(
            """
            <script>
              try { (window.parent || window).scrollTo({ top: 0, left: 0, behavior: 'smooth' }); }
              catch(e) { window.scrollTo(0,0); }
            </script>
            """,
            height=0,
            width=0,
        )

    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 100px !important;
                padding-left: 300px !important;
                padding-bottom: 300px !important;
                background-image: url('https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/main/assets/Teams%20Background%20Tablet_EN.png?raw=true');
                background-repeat: no-repeat;
                background-size: cover;
                background-position: center top;
                color: white;
            }
            .main-section { margin-left: 200px; max-width: 820px; text-align: left; }
            .main-title { font-size: 42px; font-weight: 800; margin-bottom: 16px; }
            .context { font-size: 20px; line-height: 1.6; margin-top: 8px; margin-bottom: 36px; opacity: 0.95; max-width: 700px; text-align: left; }
            .single-button { display: flex; flex-direction: column; gap: 16px; }

            div.stButton > button {
                background: linear-gradient(90deg, rgba(255,255,255,0.22), rgba(255,255,255,0.10)) !important;
                color: #ffffff !important;
                border: 2px solid rgba(255, 255, 255, 0.8) !important;
                font-size: 32px !important; font-weight: 800 !important;
                padding: 28px 36px !important; width: 480px !important; min-height: 92px !important;
                border-radius: 16px !important; text-align: left !important; backdrop-filter: blur(3px);
                box-shadow: 0 10px 28px rgba(0,0,0,0.35) !important;
            }
            div.stButton > button:hover {
                background: rgba(255,255,255,0.28) !important;
                border-color: #ffffff !important;
                box-shadow: 0 14px 36px rgba(0,0,0,0.45) !important;
                transform: translateY(-1px);
            }
            .main-section a { color: #fff !important; text-decoration: underline; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='main-section'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='main-title'>Welcome to the AI-powered Explorer of the Public Service Employee Survey (PSES)</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='context' style='max-width:680px; width:100%;'>
        <p>
        The <strong>PSES AI Explorer</strong> is an interactive tool designed to help users navigate, analyze, and interpret the
        <a href='https://www.canada.ca/en/treasury-board-secretariat/services/innovation/public-service-employee-survey.html' target='_blank'>Public Service Employee Survey (PSES)</a>
        results from <strong>2019 to 2024</strong>. It combines open data with AI-assisted insights to help identify trends, challenges, and opportunities for action across the federal public service.
        </p>
        <p>
        A key feature of the PSES AI Explorer is its <strong>AI-powered questionnaire search</strong>. Using <strong>semantic search</strong>, the system goes beyond simple keyword matching to understand the meaning and context of a query. It recognizes related concepts and phrasing, helping you find questions that reflect the same ideas even when the wording differs.
        </p>
        <p>
        Once a question is selected, you can explore results by <strong>year</strong> and <strong>demographic group</strong> through a simple, guided interface. Results are presented in a standardized format with summary percentages and a brief AI-generated narrative highlighting the main trends.
        </p>
        <p><em>Together, these features make the PSES AI Explorer a powerful, user-friendly platform for transforming survey data into actionable insights.</em></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='single-button'>", unsafe_allow_html=True)
    if st.button("▶️ Start your search", key="menu_start_button"):
        st.session_state["menu1_mount_nonce"] = time.time()
        goto("menu1")
    st.markdown("</div>", unsafe_allow_html=True)


# ── Menu 1 ───────────────────────────────────────────────────────────────────
def render_menu1():
    _clear_bg_css()

    # 1) Render Menu 1 FIRST so the query page appears immediately
    try:
        from menu1.main import run_menu1
        run_menu1()
    except Exception as e:
        st.error(f"Menu 1 is unavailable: {type(e).__name__}: {e}")
        st.stop()

    # 2) After Menu 1 has rendered, do the heavy load with a centered overlay spinner
    #    Then rerun so the overlay disappears and never remains on screen.
    if not st.session_state.get("_menu1_backend_ready", False):
        if not _dl:
            st.error(f"Data loader not ready: {_loader_err or 'utils.data_loader failed to import.'}")
            st.stop()

        _show_center_overlay("Loading database into memory…")

        # Run warmup hooks (only if present)
        try:
            if callable(prewarm_all):
                prewarm_all()
            if callable(preload_pswide_dataframe):
                preload_pswide_dataframe()
        finally:
            st.session_state["_menu1_backend_ready"] = True

        # Clear overlay by rerunning after backend is ready
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    st.markdown("---")
    if st.button("🔙 Return to Home Page", key="back_home_btn"):
        st.session_state["_scroll_top_home"] = True
        goto("home")


# ── Entry ────────────────────────────────────────────────────────────────────
def main():
    if "run_menu" in st.session_state:
        st.session_state.pop("run_menu")

    if "_nav" not in st.session_state:
        st.session_state["_nav"] = "home"

    page = st.session_state["_nav"]
    if page == "menu1":
        render_menu1()
    else:
        render_home()


if __name__ == "__main__":
    main()
