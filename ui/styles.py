"""
ui/styles.py
===============================================================================
The application's visual system: CSS injected once at startup. Aims for a
clean, professional AI-workspace look -- cards, whitespace, restrained
color, subtle borders -- rather than a default Streamlit prototype look.
===============================================================================
"""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
:root {
    --accent: #4f46e5;
    --accent-soft: rgba(79, 70, 229, 0.10);
    --border: rgba(120, 120, 130, 0.18);
    --text-muted: #8a8f98;
    --radius: 14px;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    max-width: 1180px;
}

/* ---- Typography ------------------------------------------------------- */
h1, h2, h3 { letter-spacing: -0.01em; }
.app-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.1rem; }
.app-subtitle { color: var(--text-muted); font-size: 1.05rem; margin-bottom: 1.6rem; }
.section-label {
    text-transform: uppercase; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 0.4rem;
}

/* ---- Cards -------------------------------------------------------------- */
.card {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    background: rgba(127, 127, 127, 0.02);
}
.card-hero {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.2rem 2rem;
    margin-bottom: 1.6rem;
    background: linear-gradient(180deg, var(--accent-soft), rgba(127,127,127,0.01));
}
.feature-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    height: 100%;
}
.feature-card .f-title { font-weight: 600; font-size: 0.95rem; margin-bottom: 0.25rem; }
.feature-card .f-desc { color: var(--text-muted); font-size: 0.85rem; }

/* ---- Badges / pills ------------------------------------------------------ */
.badge {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; margin-right: 0.35rem; margin-bottom: 0.3rem;
}
.badge-accent { background: var(--accent-soft); color: var(--accent); }
.badge-success { background: rgba(16, 185, 129, 0.12); color: #059669; }
.badge-warning { background: rgba(245, 158, 11, 0.14); color: #b45309; }
.badge-danger  { background: rgba(239, 68, 68, 0.12); color: #dc2626; }
.badge-neutral { background: rgba(120,120,130,0.12); color: var(--text-muted); }

.pill {
    display: inline-block; padding: 0.22rem 0.75rem; border-radius: 999px;
    font-size: 0.78rem; font-weight: 500; margin-right: 0.35rem; margin-bottom: 0.35rem;
    background: rgba(120,120,130,0.10); color: inherit;
}

/* ---- Metric tiles --------------------------------------------------------- */
.metric-tile { border: 1px solid var(--border); border-radius: 12px; padding: 0.9rem 1.1rem; }
.metric-tile .m-value { font-size: 1.5rem; font-weight: 700; line-height: 1.1; }
.metric-tile .m-label { color: var(--text-muted); font-size: 0.78rem; margin-top: 0.25rem; }

/* ---- Timeline ------------------------------------------------------------- */
.timeline-item { display: flex; align-items: flex-start; gap: 0.6rem; padding: 0.35rem 0; font-size: 0.92rem; }
.timeline-dot { font-size: 0.9rem; line-height: 1.5rem; }

/* ---- Muted text ------------------------------------------------------------ */
.muted { color: var(--text-muted); font-size: 0.88rem; }
.empty-state { text-align: center; padding: 2.4rem 1rem; color: var(--text-muted); }
.empty-state .e-icon { font-size: 2rem; margin-bottom: 0.5rem; }

/* ---- Chat ------------------------------------------------------------------ */
.source-chip {
    border: 1px solid var(--border); border-radius: 10px; padding: 0.55rem 0.75rem;
    margin-bottom: 0.5rem; font-size: 0.85rem;
}
.source-chip .s-meta { color: var(--accent); font-weight: 600; font-size: 0.78rem; margin-bottom: 0.2rem; }

/* ---- Misc --------------------------------------------------------------------- */
div[data-testid="stExpander"] { border-radius: 10px; border: 1px solid var(--border); }
[data-testid="stMetricValue"] { font-size: 1.4rem; }
hr { border-color: var(--border); }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
