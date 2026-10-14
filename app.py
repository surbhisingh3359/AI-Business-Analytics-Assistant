"""Streamlit interface for the AI Business Analytics Assistant.

The interface stays a thin presentation layer around the deterministic pipeline
in ``src``. It is arranged as an app shell: a floating dock switches between
views, the sidebar owns the dataset and the run controls, and every expensive
step is explicit and stored in session state so ordinary reruns never retrain
a model.

The dock is a CSS interpretation of the React Bits Dock component
(https://reactbits.dev/components/dock). The geometry constants below mirror
that component's defaults. React Bits maps a continuous cursor distance onto a
sprung item size; Streamlit cannot run that motion-value maths inside the main
document, so magnification falls off across two neighbour rings instead, driven
by ``:has()`` sibling selectors and a spring-like easing curve.
"""

from __future__ import annotations

import hashlib
import html
import io
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import streamlit as st

from src.business_analysis_pipeline import run_business_analysis
from src.predictive_analysis import (
    can_run_prediction,
    detect_prediction_task,
    detect_prediction_targets,
    run_prediction,
)

from ui.components import metric_card, section_header, status_badge

APP_TITLE = "AI Business Analytics Copilot"
APP_TAGLINE = "From raw data to insights, predictions, and decisions"
PREDICTION_DISPLAY_LIMIT = 500
MAX_CLASSIFICATION_CLASSES = 30
PREVIEW_ROW_LIMIT = 250

# Dock geometry in pixels, mirroring the React Bits Dock defaults.
DOCK_BASE_ITEM_SIZE = 50
DOCK_MAGNIFICATION = 70
DOCK_NEIGHBOUR_SIZES = (60, 54)
DOCK_PANEL_HEIGHT = 68
# Spring-like easing standing in for { mass: 0.1, stiffness: 150, damping: 12 }.
DOCK_EASING = "cubic-bezier(0.16, 1, 0.3, 1)"
DOCK_DURATION_MS = 260
DOCK_LABEL_DURATION_MS = 200
DOCK_LABEL_RISE_PX = 10

# Requirement gates: None is always reachable, "dataset" needs an upload,
# "analysis" needs a completed deterministic run for the active dataset.
NAV_ITEMS: tuple[dict[str, Any], ...] = (
    {
        "view": "overview",
        "label": "Executive Overview",
        "icon": ":material/space_dashboard:",
        "requires": None,
    },
    {
        "view": "data",
        "label": "Data Health",
        "icon": ":material/table_chart:",
        "requires": "dataset",
    },
    {
        "view": "descriptive",
        "label": "What Happened?",
        "icon": ":material/insights:",
        "requires": "analysis",
    },
    {
        "view": "predictive",
        "label": "What's Next?",
        "icon": ":material/trending_up:",
        "requires": "dataset",
    },
    {
        "view": "decisions",
        "label": "What Should We Do?",
        "icon": ":material/checklist:",
        "requires": "analysis",
    },
    {
        "view": "technical",
        "label": "Technical",
        "icon": ":material/terminal:",
        "requires": "analysis",
    },
)

VIEW_IDS: tuple[str, ...] = tuple(item["view"] for item in NAV_ITEMS)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """Create the small, explicit state surface used by this UI."""

    defaults: dict[str, Any] = {
        "view": "overview",
        "view_switches": 0,
        "scroll_pending": False,
        "dataset_signature": None,
        "analysis_result": None,
        "analysis_signature": None,
        "analysis_options": None,
        "analysis_requested": False,
        "analysis_error": None,
        "predictive_results": {},
        "prediction_errors": {},
        "prediction_busy": False,
        "max_insights": 5,
        "max_decisions": 5,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_view(view: str) -> None:
    """Switch the active view without reloading the page."""

    if view in VIEW_IDS and view != st.session_state.view:
        st.session_state.view = view
        st.session_state.view_switches += 1
        st.session_state.scroll_pending = True


def scroll_main_to_top() -> None:
    """Return the reading position to the top after a dock switch.

    Streamlit scrolls ``section.stMain`` rather than the window and there is no
    Python API for it. The nonce keeps the element unique so it runs on every
    switch.
    """

    st.html(
        f"""
        <script>
            /* switch {st.session_state.view_switches} */
            const main = document.querySelector('section.stMain');
            if (main) main.scrollTo({{top: 0, behavior: 'smooth'}});
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def request_analysis() -> None:
    """Queue a deterministic analysis run for the current script pass."""

    st.session_state.analysis_requested = True


def clear_dataset_results() -> None:
    """Forget results tied to a previous uploaded dataset."""

    st.session_state.analysis_result = None
    st.session_state.analysis_signature = None
    st.session_state.analysis_options = None
    st.session_state.analysis_requested = False
    st.session_state.analysis_error = None
    st.session_state.predictive_results = {}
    st.session_state.prediction_errors = {}
    st.session_state.prediction_busy = False


def dataset_signature(file_bytes: bytes) -> str:
    """Return a stable identifier for the exact uploaded CSV contents."""

    return hashlib.sha256(file_bytes).hexdigest()


def prediction_key(signature: str, target: str) -> str:
    """Build a state key for one dataset and target combination."""

    return f"{signature}:{target}"


# ---------------------------------------------------------------------------
# Dataset profiling
# ---------------------------------------------------------------------------


def is_identifier_column(column: Any) -> bool:
    """Return whether a column name is likely an identifier."""

    name = str(column).strip().lower()
    return name == "id" or name.endswith("_id")


def is_categorical_series(series: pd.Series) -> bool:
    """Return whether a Series should be treated as categorical."""

    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def build_dataset_profile(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Build a display-oriented profile once per uploaded file."""

    rows = int(len(dataframe))
    columns = int(len(dataframe.columns))
    missing_by_column = dataframe.isna().sum()
    missing_cells = int(missing_by_column.sum())
    total_cells = rows * columns

    quality_rows: list[dict[str, Any]] = []
    for column in dataframe.columns:
        series = dataframe[column]
        missing = int(missing_by_column[column])
        try:
            unique = int(series.nunique(dropna=True))
        except TypeError:
            unique = 0

        quality_rows.append(
            {
                "Column": str(column),
                "Type": str(series.dtype),
                "Missing": missing,
                "Missing %": round(missing / rows * 100, 1) if rows else 0.0,
                "Unique": unique,
            }
        )

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column
        for column in dataframe.columns
        if is_categorical_series(dataframe[column])
    ]
    datetime_columns = [
        column
        for column in dataframe.columns
        if pd.api.types.is_datetime64_any_dtype(dataframe[column])
    ]

    return {
        "rows": rows,
        "columns": columns,
        "missing_cells": missing_cells,
        "missing_rate": (
            float(missing_cells / total_cells * 100) if total_cells else 0.0
        ),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "memory_mb": float(
            dataframe.memory_usage(deep=True).sum() / (1024 * 1024)
        ),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "identifier_columns": [
            column for column in dataframe.columns if is_identifier_column(column)
        ],
        "quality_rows": quality_rows,
    }


@st.cache_data(show_spinner=False)
def load_dataset_bundle(
    file_bytes: bytes,
    file_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse a supported upload and compute the reusable profile."""

    extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    buffer = io.BytesIO(file_bytes)

    if extension == "csv":
        dataframe = pd.read_csv(buffer)
    elif extension == "xlsx":
        dataframe = pd.read_excel(buffer)
    else:
        raise ValueError("Upload a CSV or XLSX file.")

    if dataframe.empty:
        raise ValueError("The uploaded file does not contain any data rows.")
    return dataframe, build_dataset_profile(dataframe)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def safe_text(value: Any, fallback: str = "—") -> str:
    """Escape a value before placing it in trusted application HTML."""

    if value is None:
        return fallback
    text = str(value).strip()
    return html.escape(text) if text else fallback


def html_list(values: Any, fallback: str = "None") -> str:
    """Format a short list safely for inline HTML."""

    if not isinstance(values, (list, tuple, set)):
        values = [values] if values else []
    items = [safe_text(item) for item in values if str(item).strip()]
    return " · ".join(items) if items else fallback


def numeric_value(value: Any) -> float | None:
    """Return a finite float when possible."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_number(value: Any, decimals: int = 2, fallback: str = "—") -> str:
    """Render a number without displaying NaN or Infinity."""

    number = numeric_value(value)
    return f"{number:,.{decimals}f}" if number is not None else fallback


def format_percent(value: Any, decimals: int = 1, fallback: str = "—") -> str:
    """Render a fractional metric as a percentage."""

    number = numeric_value(value)
    return f"{number * 100:.{decimals}f}%" if number is not None else fallback


def render_html(markup: str) -> None:
    """Render known-safe application markup."""

    st.html(markup)


# ---------------------------------------------------------------------------
# Theme and styling
# ---------------------------------------------------------------------------


def light_palette() -> dict[str, str]:
    """Return the light surface palette, matching ``[theme.light]`` in config."""

    return {
        "bg": "#f4f6fc",
        "glow_one": "rgba(99, 102, 241, 0.22)",
        "glow_two": "rgba(56, 189, 248, 0.18)",
        "surface": "rgba(255, 255, 255, 0.86)",
        "surface_strong": "rgba(255, 255, 255, 0.96)",
        "surface_alt": "rgba(233, 238, 250, 0.85)",
        "border": "rgba(15, 23, 42, 0.10)",
        "border_strong": "rgba(15, 23, 42, 0.16)",
        "text": "#101a30",
        "muted": "#5a6a86",
        "accent": "#3b5bdb",
        "accent_alt": "#7c3aed",
        "accent_soft": "rgba(59, 91, 219, 0.12)",
        "success": "#0f8a5f",
        "warning": "#a35c05",
        "danger": "#c02a49",
        "shadow": "0 20px 50px rgba(23, 43, 84, 0.12)",
        "dock_bg": "rgba(255, 255, 255, 0.72)",
        "dock_shadow": "0 22px 48px rgba(23, 43, 84, 0.20)",
    }


def dark_palette() -> dict[str, str]:
    """Return the dark surface palette, matching ``[theme.dark]`` in config."""

    return {
        "bg": "#05070f",
        "glow_one": "rgba(124, 58, 237, 0.30)",
        "glow_two": "rgba(29, 78, 216, 0.26)",
        "surface": "rgba(255, 255, 255, 0.045)",
        "surface_strong": "rgba(255, 255, 255, 0.075)",
        "surface_alt": "rgba(255, 255, 255, 0.02)",
        "border": "rgba(255, 255, 255, 0.10)",
        "border_strong": "rgba(255, 255, 255, 0.18)",
        "text": "#eef2ff",
        "muted": "#93a0bd",
        "accent": "#8ba7ff",
        "accent_alt": "#c08bff",
        "accent_soft": "rgba(139, 167, 255, 0.14)",
        "success": "#4ade80",
        "warning": "#fbbf24",
        "danger": "#fb7185",
        "shadow": "0 24px 60px rgba(0, 0, 0, 0.45)",
        "dock_bg": "rgba(12, 16, 30, 0.72)",
        "dock_shadow": "0 22px 56px rgba(0, 0, 0, 0.55)",
    }


def palette_variables(palette: Mapping[str, str]) -> str:
    """Emit a palette as CSS custom properties."""

    names = {
        "bg": "--app-bg",
        "glow_one": "--glow-one",
        "glow_two": "--glow-two",
        "surface": "--surface",
        "surface_strong": "--surface-strong",
        "surface_alt": "--surface-alt",
        "border": "--border",
        "border_strong": "--border-strong",
        "text": "--text",
        "muted": "--muted",
        "accent": "--accent",
        "accent_alt": "--accent-alt",
        "accent_soft": "--accent-soft",
        "success": "--success",
        "warning": "--warning",
        "danger": "--danger",
        "shadow": "--shadow",
        "dock_bg": "--dock-bg",
        "dock_shadow": "--dock-shadow",
    }
    return "\n".join(
        f"                {names[key]}: {value};"
        for key, value in palette.items()
        if key in names
    )


def dock_styles() -> str:
    """Build the dock CSS from the React Bits geometry constants.

    Each dock button lives in its own Streamlit element container, so the
    containers are siblings. ``:has()`` turns that sibling chain into a
    distance approximation: the hovered item reaches ``DOCK_MAGNIFICATION``
    and each ring outwards settles back toward ``DOCK_BASE_ITEM_SIZE``.
    """

    item = '[data-testid="stElementContainer"]'
    dock = ".st-key-dock"

    base = DOCK_BASE_ITEM_SIZE
    ring_one, ring_two = DOCK_NEIGHBOUR_SIZES
    scales = {
        "hovered": DOCK_MAGNIFICATION / base,
        "ring_one": ring_one / base,
        "ring_two": ring_two / base,
    }
    # Lift each item by half of the size it gained, so the row keeps a
    # common baseline the way a macOS dock does.
    lifts = {
        "hovered": (DOCK_MAGNIFICATION - base) / 2,
        "ring_one": (ring_one - base) / 2,
        "ring_two": (ring_two - base) / 2,
    }
    headroom = round((DOCK_MAGNIFICATION - base) * 1.5)
    label_scale = base / DOCK_MAGNIFICATION

    def magnify(name: str) -> str:
        return (
            f"transform: translateY(-{lifts[name]:.0f}px) "
            f"scale({scales[name]:.3f});"
        )

    return f"""
        /* ---- dock panel ---------------------------------------------- */
        {dock} {{
            position: fixed !important;
            left: 50%;
            bottom: 1.15rem;
            transform: translateX(-50%);
            z-index: 1000;
            width: max-content;
            max-width: calc(100vw - 1.5rem);
            min-height: {DOCK_PANEL_HEIGHT}px;
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: flex-end !important;
            justify-content: center;
            gap: 0.25rem !important;
            padding: 0.5rem 0.6rem;
            border: 1px solid var(--border-strong);
            border-radius: 1.4rem;
            background: var(--dock-bg);
            box-shadow: var(--dock-shadow);
            backdrop-filter: blur(22px) saturate(150%);
            -webkit-backdrop-filter: blur(22px) saturate(150%);
            overflow: visible !important;
            transition: padding-top {DOCK_DURATION_MS}ms {DOCK_EASING};
        }}

        {dock}:has(button:hover), {dock}:has(button:focus-visible) {{
            padding-top: {headroom}px;
        }}

        {dock} > div {{
            display: flex;
            align-items: flex-end;
            justify-content: center;
            width: auto !important;
            min-width: 0 !important;
            flex: 0 0 auto !important;
            overflow: visible !important;
        }}

        {dock} {item} {{
            width: auto !important;
            min-width: 0 !important;
            flex: 0 0 auto !important;
            overflow: visible !important;
        }}

        {dock} .stButton, {dock} [data-testid="stButton"] {{
            overflow: visible !important;
        }}

        /* ---- dock items ---------------------------------------------- */
        {dock} button {{
            position: relative;
            display: grid;
            place-items: center;
            width: {base}px !important;
            height: {base}px !important;
            min-height: {base}px !important;
            padding: 0 !important;
            border: 1px solid transparent !important;
            border-radius: 0.9rem !important;
            background: var(--surface) !important;
            color: var(--muted) !important;
            overflow: visible !important;
            transform-origin: bottom center;
            transition:
                transform {DOCK_DURATION_MS}ms {DOCK_EASING},
                background {DOCK_DURATION_MS}ms ease,
                color {DOCK_DURATION_MS}ms ease,
                border-color {DOCK_DURATION_MS}ms ease;
        }}

        {dock} button [data-testid="stIconMaterial"],
        {dock} button span[class*="material"] {{
            font-size: 1.35rem !important;
            width: auto !important;
            margin: 0 !important;
            line-height: 1 !important;
        }}

        /* ---- magnification rings, widest ring first ------------------ */
        {dock} {item}:has(+ {item} + {item} button:hover) button,
        {dock} {item}:has(button:hover) + {item} + {item} button,
        {dock} {item}:has(+ {item} + {item} button:focus-visible) button,
        {dock} {item}:has(button:focus-visible) + {item} + {item} button {{
            {magnify("ring_two")}
        }}

        {dock} {item}:has(+ {item} button:hover) button,
        {dock} {item}:has(button:hover) + {item} button,
        {dock} {item}:has(+ {item} button:focus-visible) button,
        {dock} {item}:has(button:focus-visible) + {item} button {{
            {magnify("ring_one")}
            color: var(--text) !important;
        }}

        {dock} {item}:has(button:hover) button,
        {dock} {item}:has(button:focus-visible) button {{
            {magnify("hovered")}
            background: var(--accent-soft) !important;
            border-color: var(--border-strong) !important;
            color: var(--text) !important;
        }}

        {dock} button:disabled {{
            opacity: 0.34;
            cursor: not-allowed;
        }}

        /* ---- active item, macOS style running indicator -------------- */
        {dock} button[kind="primary"],
        {dock} button[data-testid="stBaseButton-primary"] {{
            background: linear-gradient(
                140deg, var(--accent), var(--accent-alt)
            ) !important;
            border-color: transparent !important;
            color: #ffffff !important;
            box-shadow: 0 10px 24px var(--accent-soft);
        }}

        {dock} button[kind="primary"]::after,
        {dock} button[data-testid="stBaseButton-primary"]::after {{
            content: "";
            position: absolute;
            bottom: -0.42rem;
            left: 50%;
            width: 4px;
            height: 4px;
            border-radius: 50%;
            background: var(--accent);
            transform: translateX(-50%);
        }}

        /* ---- dock label, the React Bits tooltip ---------------------- */
        {dock} button [data-testid="stMarkdownContainer"] {{
            position: absolute;
            bottom: calc(100% + 0.6rem);
            left: 50%;
            width: max-content !important;
            /* The label lives inside the magnified button, so it is scaled
               back down to keep a constant chip size like the original. */
            transform: translate(-50%, 0) scale({label_scale:.4f});
            padding: 0.26rem 0.5rem;
            border: 1px solid var(--border-strong);
            border-radius: 0.5rem;
            background: var(--surface-strong);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            color: var(--text);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition:
                opacity {DOCK_LABEL_DURATION_MS}ms ease,
                transform {DOCK_LABEL_DURATION_MS}ms {DOCK_EASING};
        }}

        {dock} button [data-testid="stMarkdownContainer"] p {{
            margin: 0 !important;
            font-size: inherit !important;
            font-weight: inherit !important;
            line-height: 1.2 !important;
            color: inherit !important;
        }}

        {dock} {item}:has(button:hover) button [data-testid="stMarkdownContainer"],
        {dock} {item}:has(button:focus-visible) button
            [data-testid="stMarkdownContainer"] {{
            opacity: 1;
            transform: translate(-50%, -{DOCK_LABEL_RISE_PX}px)
                scale({label_scale:.4f});
        }}

        @media (max-width: 640px) {{
            {dock} {{
                bottom: 0.65rem;
                padding: 0.4rem 0.45rem;
                gap: 0.1rem !important;
            }}

            {dock} button {{
                width: 2.6rem !important;
                height: 2.6rem !important;
                min-height: 2.6rem !important;
            }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            {dock}, {dock} button,
            {dock} button [data-testid="stMarkdownContainer"] {{
                transition: none !important;
            }}

            {dock} {item}:has(button:hover) button {{
                transform: none;
            }}
        }}
    """


def render_theme_bridge() -> None:
    """Mirror Streamlit's resolved theme onto ``html[data-app-theme]``.

    Streamlit publishes no theme marker in the DOM, repaints its own chrome
    from an injected stylesheet rather than an attribute, and does not rerun
    the script when the theme changes, so ``st.context.theme`` is always one
    run behind. Reading back the background colour Streamlit actually painted
    is the only signal that stays in step with its widgets, dataframes, and
    charts whether the change came from the operating system or from
    Settings › Appearance. The stylesheet's ``prefers-color-scheme`` block
    covers the case where this snippet cannot run.
    """

    st.html(
        """
        <script>
            (() => {
                if (window.__appThemeBridge) return;
                window.__appThemeBridge = true;

                const brightness = (colour) => {
                    const parts = (colour || '').match(/[-0-9.]+/g);
                    if (!parts || parts.length < 3) return null;
                    if (parts.length > 3 && Number(parts[3]) === 0) return null;
                    const [r, g, b] = parts.slice(0, 3).map(Number);
                    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
                };

                const sync = () => {
                    const root = document.documentElement;
                    const level =
                        brightness(getComputedStyle(document.body).backgroundColor) ??
                        brightness(getComputedStyle(root).backgroundColor);
                    if (level === null || level === undefined) return;
                    const theme = level > 0.5 ? 'light' : 'dark';
                    if (root.dataset.appTheme !== theme) {
                        root.dataset.appTheme = theme;
                    }
                };

                sync();
                window.setInterval(sync, 400);
            })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def render_styles() -> None:
    """Install the application styling, including the floating dock.

    Both palettes ship in the stylesheet so the surface can follow Streamlit's
    chrome without waiting for a rerun. ``prefers-color-scheme`` handles the
    browser or operating system preference, and the ``data-app-theme``
    attribute set by :func:`render_theme_bridge` takes precedence so a choice
    made in Settings › Appearance is honoured too.
    """

    render_html(
        f"""
        <style>
            :root {{
{palette_variables(dark_palette())}
            }}

            @media (prefers-color-scheme: light) {{
                :root {{
{palette_variables(light_palette())}
                }}
            }}

            html[data-app-theme="dark"] {{
{palette_variables(dark_palette())}
            }}

            html[data-app-theme="light"] {{
{palette_variables(light_palette())}
            }}

            .stApp {{
                background:
                    radial-gradient(
                        60rem 32rem at 12% -8%, var(--glow-one), transparent 70%
                    ),
                    radial-gradient(
                        52rem 30rem at 92% 4%, var(--glow-two), transparent 70%
                    ),
                    var(--app-bg);
                color: var(--text);
            }}

            [data-testid="stHeader"] {{
                background: transparent;
            }}

            [data-testid="stSidebar"] {{
                background: var(--surface-alt);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border-right: 1px solid var(--border);
            }}

            [data-testid="stSidebar"] * {{
                color: var(--text);
            }}

            .block-container {{
                max-width: 1360px;
                padding-top: 1.5rem;
                padding-bottom: 9rem;
            }}

            h1, h2, h3, h4, h5, h6, p, li, label {{
                color: var(--text);
            }}

            /* ---- Streamlit primitives ------------------------------- */
            [data-testid="stMetric"],
            [data-testid="stExpander"] details,
            [data-testid="stFileUploaderDropzone"],
            [data-testid="stDataFrame"] {{
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 1rem;
            }}

            [data-testid="stMetric"] {{
                padding: 0.95rem 1rem;
                box-shadow: var(--shadow);
            }}

            [data-testid="stMetricLabel"] {{
                color: var(--muted);
            }}

            [data-testid="stExpander"] details {{
                overflow: hidden;
            }}

            [data-testid="stExpander"] summary {{
                color: var(--text);
                font-weight: 650;
            }}

            .stMain [data-testid="stBaseButton-primary"] {{
                background: linear-gradient(
                    135deg, var(--accent), var(--accent-alt)
                );
                border: none;
                color: #ffffff;
                font-weight: 700;
                box-shadow: 0 14px 32px var(--accent-soft);
                transition: transform 200ms {DOCK_EASING};
            }}

            .stMain [data-testid="stBaseButton-primary"]:hover {{
                transform: translateY(-1px);
            }}

            [data-testid="stStatusWidget"], [data-testid="stStatus"] {{
                border-radius: 1rem;
            }}

            /* ---- shell ---------------------------------------------- */
            .topbar {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                margin-bottom: 2rem;
            }}

            .brand {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }}

            .brand-mark {{
                width: 2.6rem;
                height: 2.6rem;
                display: grid;
                place-items: center;
                border-radius: 0.85rem;
                color: #ffffff;
                background: linear-gradient(
                    135deg, var(--accent), var(--accent-alt)
                );
                font-size: 1.05rem;
                font-weight: 850;
                box-shadow: 0 12px 28px var(--accent-soft);
            }}

            .brand-name {{
                font-size: 1.02rem;
                font-weight: 800;
                letter-spacing: -0.025em;
                line-height: 1.15;
            }}

            .brand-copy {{
                color: var(--muted);
                font-size: 0.76rem;
                margin-top: 0.1rem;
            }}

            .shell-chip {{
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                border: 1px solid var(--border);
                border-radius: 999px;
                background: var(--surface);
                color: var(--muted);
                padding: 0.34rem 0.7rem;
                font-size: 0.72rem;
                font-weight: 650;
            }}

            .shell-chip b {{
                color: var(--text);
                font-weight: 750;
            }}

            /* ---- hero ----------------------------------------------- */
            .hero {{
                max-width: 900px;
                margin: 0 auto 2.4rem;
                text-align: center;
                animation: rise 520ms {DOCK_EASING} both;
            }}

            .eyebrow {{
                color: var(--accent);
                font-size: 0.71rem;
                font-weight: 800;
                letter-spacing: 0.16em;
                text-transform: uppercase;
                margin-bottom: 0.75rem;
            }}

            .hero h1 {{
                margin: 0;
                font-size: clamp(2.4rem, 5.6vw, 4.6rem);
                line-height: 0.98;
                letter-spacing: -0.06em;
                background: linear-gradient(
                    120deg, var(--text) 30%, var(--accent) 70%, var(--accent-alt)
                );
                -webkit-background-clip: text;
                background-clip: text;
                -webkit-text-fill-color: transparent;
            }}

            .hero p {{
                max-width: 660px;
                margin: 1.1rem auto 0;
                color: var(--muted);
                font-size: 1rem;
                line-height: 1.7;
            }}

            .journey {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 0.4rem;
                margin-top: 1.3rem;
            }}

            .journey span {{
                border: 1px solid var(--border);
                border-radius: 999px;
                background: var(--surface);
                color: var(--muted);
                padding: 0.34rem 0.68rem;
                font-size: 0.74rem;
                font-weight: 650;
            }}

            /* ---- sections and panels -------------------------------- */
            .section-header {{
                position: relative;
                margin: 0 0 1.2rem;
                padding-left: 0.9rem;
            }}

            .section-header::before {{
                content: "";
                position: absolute;
                left: 0;
                top: 0.15rem;
                bottom: 0.15rem;
                width: 3px;
                border-radius: 999px;
                background: linear-gradient(
                    var(--accent), var(--accent-alt)
                );
            }}

            .section-header h2 {{
                margin: 0.1rem 0 0;
                font-size: 1.65rem;
                letter-spacing: -0.045em;
            }}

            .section-header p {{
                max-width: 720px;
                margin: 0.35rem 0 0;
                color: var(--muted);
                line-height: 1.65;
            }}

            .panel {{
                height: 100%;
                box-sizing: border-box;
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 1.15rem;
                padding: 1.05rem 1.1rem;
                box-shadow: var(--shadow);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                transition:
                    transform 260ms {DOCK_EASING},
                    border-color 260ms ease;
                animation: rise 420ms {DOCK_EASING} both;
            }}

            .panel:hover {{
                transform: translateY(-2px);
                border-color: var(--border-strong);
            }}

            .panel-label {{
                color: var(--muted);
                font-size: 0.7rem;
                font-weight: 800;
                letter-spacing: 0.1em;
                text-transform: uppercase;
            }}

            .panel-value {{
                font-size: 1.62rem;
                line-height: 1.1;
                font-weight: 820;
                letter-spacing: -0.05em;
                margin-top: 0.45rem;
                overflow-wrap: anywhere;
            }}

            /* Text values such as file names need to wrap, not to shout. */
            .panel-value-text {{
                font-size: 1.05rem;
                font-weight: 760;
                letter-spacing: -0.02em;
                line-height: 1.35;
            }}

            [data-testid="stVegaLiteChart"],
            [data-testid="stArrowVegaLiteChart"] {{
                border: 1px solid var(--border);
                border-radius: 1.05rem;
                overflow: hidden;
            }}

            .panel-note {{
                color: var(--muted);
                font-size: 0.8rem;
                line-height: 1.5;
                margin-top: 0.35rem;
            }}

            .notice {{
                border-left: 3px solid var(--accent);
            }}

            .notice-success {{
                border-left-color: var(--success);
            }}

            .notice-warning {{
                border-left-color: var(--warning);
            }}

            .notice-danger {{
                border-left-color: var(--danger);
            }}

            .notice-title {{
                font-size: 1rem;
                font-weight: 780;
                margin-top: 0.35rem;
            }}

            .notice-copy {{
                color: var(--muted);
                line-height: 1.6;
                margin: 0.4rem 0 0;
            }}

            /* ---- insight and decision cards ------------------------- */
            .insight, .decision {{
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 1.1rem;
                padding: 1rem 1.1rem;
                margin: 0.7rem 0;
                box-shadow: var(--shadow);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                animation: rise 420ms {DOCK_EASING} both;
            }}

            .insight-meta, .decision-index {{
                color: var(--accent);
                font-size: 0.7rem;
                font-weight: 800;
                letter-spacing: 0.09em;
                text-transform: uppercase;
            }}

            .insight h3 {{
                margin: 0.38rem 0 0;
                font-size: 1rem;
                letter-spacing: -0.02em;
            }}

            .insight p {{
                margin: 0.45rem 0 0;
                color: var(--muted);
                line-height: 1.6;
            }}

            .decision {{
                background: linear-gradient(
                    140deg, var(--surface-strong), var(--surface)
                );
            }}

            .decision h3 {{
                margin: 0.42rem 0 0;
                font-size: 1.02rem;
                font-weight: 720;
                line-height: 1.5;
            }}

            .status-pill {{
                display: inline-block;
                border-radius: 999px;
                padding: 0.22rem 0.55rem;
                font-size: 0.73rem;
                font-weight: 760;
                text-transform: capitalize;
            }}

            .status-completed {{
                background: color-mix(in srgb, var(--success) 18%, transparent);
                color: var(--success);
            }}

            .status-skipped {{
                background: color-mix(in srgb, var(--warning) 18%, transparent);
                color: var(--warning);
            }}

            .status-failed {{
                background: color-mix(in srgb, var(--danger) 18%, transparent);
                color: var(--danger);
            }}

            .footer {{
                margin-top: 2.8rem;
                color: var(--muted);
                font-size: 0.76rem;
                text-align: center;
            }}

            @keyframes rise {{
                from {{
                    opacity: 0;
                    transform: translateY(8px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}

            @media (prefers-reduced-motion: reduce) {{
                .hero, .panel, .insight, .decision {{
                    animation: none !important;
                }}
            }}

            @media (max-width: 640px) {{
                .block-container {{
                    padding-left: 1rem;
                    padding-right: 1rem;
                    padding-bottom: 8rem;
                }}

                .hero h1 {{
                    font-size: 2.6rem;
                }}
            }}

            {dock_styles()}
        </style>
        """
    )


# ---------------------------------------------------------------------------
# Shared UI atoms
# ---------------------------------------------------------------------------


def render_section(eyebrow: str, title: str, copy: str) -> None:
    """Render a consistently styled view header."""

    render_html(
        f"""
        <div class="section-header">
            <div class="eyebrow">{safe_text(eyebrow)}</div>
            <h2>{safe_text(title)}</h2>
            <p>{safe_text(copy)}</p>
        </div>
        """
    )


def render_metric_card(
    label: str,
    value: str,
    note: str = "",
    text_value: bool = False,
) -> None:
    """Render one lightweight, visually consistent metric card.

    Set ``text_value`` for values that are words rather than numbers, so long
    strings such as file names wrap instead of overflowing the card.
    """

    note_markup = (
        f'<div class="panel-note">{safe_text(note)}</div>' if note else ""
    )
    value_class = "panel-value panel-value-text" if text_value else "panel-value"
    render_html(
        f"""
        <div class="panel">
            <div class="panel-label">{safe_text(label)}</div>
            <div class="{value_class}">{safe_text(value)}</div>
            {note_markup}
        </div>
        """
    )


def render_notice(
    label: str,
    title: str,
    copy: str,
    tone: str = "accent",
) -> None:
    """Render a compact explanation card with a semantic colour."""

    tone_class = {
        "success": "notice-success",
        "warning": "notice-warning",
        "danger": "notice-danger",
    }.get(tone, "")

    render_html(
        f"""
        <div class="panel notice {tone_class}">
            <div class="panel-label">{safe_text(label)}</div>
            <div class="notice-title">{safe_text(title)}</div>
            <p class="notice-copy">{safe_text(copy)}</p>
        </div>
        """
    )


def render_metric_row(cards: Sequence[Sequence[Any]]) -> None:
    """Render a row of metric cards.

    Each card is ``(label, value, note)`` with an optional fourth element that
    marks the value as text rather than a number.
    """

    if not cards:
        return
    columns = st.columns(len(cards))
    for container, card in zip(columns, cards):
        label, value, note = card[0], card[1], card[2]
        text_value = bool(card[3]) if len(card) > 3 else False
        with container:
            render_metric_card(label, value, note, text_value=text_value)


def status_pill(status: Any) -> str:
    """Build a safe status badge for execution cards."""

    normalized = str(status or "unknown").lower()
    class_name = {
        "completed": "status-completed",
        "skipped": "status-skipped",
        "failed": "status-failed",
    }.get(normalized, "status-skipped")
    return (
        f'<span class="status-pill {class_name}">'
        f"{safe_text(normalized)}</span>"
    )


# ---------------------------------------------------------------------------
# Shell: dock, top bar, sidebar, footer
# ---------------------------------------------------------------------------


def view_available(
    requires: str | None,
    has_dataset: bool,
    has_analysis: bool,
) -> bool:
    """Return whether a view's data requirement is currently satisfied."""

    if requires == "analysis":
        return has_analysis
    if requires == "dataset":
        return has_dataset
    return True


def resolve_view(has_dataset: bool, has_analysis: bool) -> str:
    """Return the active view, falling back when it is not reachable."""

    view = st.session_state.view
    requirements = {item["view"]: item["requires"] for item in NAV_ITEMS}
    if view not in requirements or not view_available(
        requirements[view], has_dataset, has_analysis
    ):
        view = "overview"
    st.session_state.view = view
    return view


def render_dock(active_view: str, has_dataset: bool, has_analysis: bool) -> None:
    """Render the floating dock navigator.

    Clicks are handled with ``on_click`` callbacks so the new view is already
    in session state by the time this row is drawn on the following pass.
    """

    with st.container(
        key="dock",
        horizontal=True,
        horizontal_alignment="center",
        gap="small",
    ):
        for item in NAV_ITEMS:
            enabled = view_available(item["requires"], has_dataset, has_analysis)
            st.button(
                item["label"],
                key=f"dock_{item['view']}",
                icon=item["icon"],
                type="primary" if item["view"] == active_view else "secondary",
                disabled=not enabled,
                on_click=set_view,
                args=(item["view"],),
            )


def topbar_markup(status_chip: str) -> str:
    """Build the brand row and a compact workspace status chip."""

    return f"""
        <div class="topbar">
            <div class="brand">
                <div class="brand-mark">◈</div>
                <div>
                    <div class="brand-name">{safe_text(APP_TITLE)}</div>
                    <div class="brand-copy">{safe_text(APP_TAGLINE)}</div>
                </div>
            </div>
            <div class="shell-chip">{status_chip}</div>
        </div>
        """


def render_sidebar() -> tuple[Any, tuple[int, int]]:
    """Render the workspace controls and return the upload and run options."""

    with st.sidebar:
        st.markdown("### Workspace")
        st.caption(
            "Light and dark are switched from the app menu, under "
            "Settings › Appearance."
        )

        st.divider()
        uploaded_file = st.file_uploader(
            "Dataset (CSV or XLSX)",
            type=["csv", "xlsx"],
            help="Uploaded files stay in this browser session.",
        )

        st.divider()
        st.markdown("**Analysis controls**")
        st.slider(
            "Maximum insights",
            min_value=1,
            max_value=10,
            key="max_insights",
            help="How many ranked findings are surfaced after analysis.",
        )
        st.slider(
            "Maximum decisions",
            min_value=1,
            max_value=10,
            key="max_decisions",
            help="How many business decisions are created from findings.",
        )

        st.button(
            "Run business analysis",
            type="primary",
            width="stretch",
            disabled=uploaded_file is None,
            on_click=request_analysis,
            key="sidebar_run_analysis",
        )
        st.button(
            "Clear results",
            width="stretch",
            disabled=st.session_state.analysis_result is None,
            on_click=clear_dataset_results,
            key="sidebar_clear_results",
        )
        st.caption(
            "Control changes apply the next time the deterministic analysis "
            "runs. Navigation and theme changes never recompute anything."
        )

    return uploaded_file, (
        int(st.session_state.max_insights),
        int(st.session_state.max_decisions),
    )


def render_footer() -> None:
    """Render the small, non-intrusive application footer."""

    render_html(
        """
        <div class="footer">
            AI Business Analytics · deterministic analytical core ·
            descriptive, predictive, and prescriptive views
        </div>
        """
    )


# ---------------------------------------------------------------------------
# Analysis execution
# ---------------------------------------------------------------------------


def execute_analysis(
    dataframe: pd.DataFrame,
    signature: str,
    options: tuple[int, int],
) -> None:
    """Run the deterministic pipeline once and store the result."""

    max_insights, max_decisions = options
    st.session_state.analysis_error = None

    try:
        with st.status("Building the analytical view…", expanded=True) as status:
            st.write("Profiling business structure and semantic roles.")
            st.write("Ranking insights, decisions, and executable analyses.")
            analysis_result = run_business_analysis(
                dataframe,
                max_insights=max_insights,
                max_decisions=max_decisions,
            )
            status.update(
                label="Business analysis complete",
                state="complete",
                expanded=False,
            )

        st.session_state.analysis_result = analysis_result
        st.session_state.analysis_signature = signature
        st.session_state.analysis_options = options
        st.session_state.predictive_results = {}
        st.session_state.prediction_errors = {}

    except (TypeError, ValueError, KeyError) as exc:
        st.session_state.analysis_result = None
        st.session_state.analysis_signature = None
        st.session_state.analysis_error = str(exc)
    except Exception as exc:  # pragma: no cover - defensive UI boundary
        st.session_state.analysis_result = None
        st.session_state.analysis_signature = None
        st.session_state.analysis_error = (
            f"The analysis stopped unexpectedly. Details: {exc}"
        )


# ---------------------------------------------------------------------------
# Predictive helpers
# ---------------------------------------------------------------------------


def target_readiness(dataframe: pd.DataFrame, target: str) -> dict[str, Any]:
    """Check UI-level readiness before a user may start a model run."""

    readiness = dict(can_run_prediction(dataframe))
    minimum_rows = int(readiness.get("minimum_rows", 10))

    if not readiness.get("available", False):
        readiness.update(
            {"ready": False, "task": None, "usable_target_rows": 0}
        )
        return readiness

    values = dataframe[target].dropna()
    usable_target_rows = int(len(values))
    readiness["usable_target_rows"] = usable_target_rows

    try:
        task = detect_prediction_task(dataframe, target)
    except (TypeError, ValueError) as exc:
        readiness.update({"ready": False, "task": None, "reason": str(exc)})
        return readiness

    readiness["task"] = task
    if usable_target_rows < minimum_rows:
        readiness.update(
            {
                "ready": False,
                "reason": (
                    f"'{target}' has only {usable_target_rows:,} usable target values. "
                    f"Choose a target with at least {minimum_rows:,} values."
                ),
            }
        )
        return readiness

    unique_values = int(values.nunique(dropna=True))
    readiness["unique_target_values"] = unique_values

    if task == "regression" and unique_values < 2:
        readiness.update(
            {
                "ready": False,
                "reason": f"'{target}' does not vary enough for regression.",
            }
        )
        return readiness

    if task == "classification":
        class_counts = values.value_counts(dropna=True)
        class_count = int(len(class_counts))
        readiness["class_count"] = class_count

        if class_count < 2:
            readiness.update(
                {
                    "ready": False,
                    "reason": (
                        f"'{target}' needs at least two outcome classes for classification."
                    ),
                }
            )
            return readiness

        if int(class_counts.min()) < 2:
            readiness.update(
                {
                    "ready": False,
                    "reason": (
                        "Each classification outcome must appear at least twice "
                        "for a safe train/test split."
                    ),
                }
            )
            return readiness

        if class_count > MAX_CLASSIFICATION_CLASSES:
            readiness.update(
                {
                    "ready": False,
                    "reason": (
                        f"'{target}' has {class_count:,} outcome classes. Choose "
                        f"a target with at most {MAX_CLASSIFICATION_CLASSES} classes."
                    ),
                }
            )
            return readiness

    readiness["ready"] = True
    readiness["reason"] = None
    return readiness


def meaningful_model_improvement(
    task: str,
    metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
) -> tuple[bool, str]:
    """Apply a business-meaningfulness threshold to baseline comparisons."""

    if task == "regression":
        model_mae = numeric_value(metrics.get("mae"))
        baseline_mae = numeric_value(baseline_metrics.get("mae"))
        model_rmse = numeric_value(metrics.get("rmse"))
        baseline_rmse = numeric_value(baseline_metrics.get("rmse"))
        model_r2 = numeric_value(metrics.get("r2"))
        baseline_r2 = numeric_value(baseline_metrics.get("r2"))

        mae_gain = (
            (baseline_mae - model_mae) / baseline_mae
            if baseline_mae and model_mae is not None
            else 0.0
        )
        rmse_gain = (
            (baseline_rmse - model_rmse) / baseline_rmse
            if baseline_rmse and model_rmse is not None
            else 0.0
        )
        r2_gain = (
            model_r2 - baseline_r2
            if model_r2 is not None and baseline_r2 is not None
            else None
        )

        if mae_gain >= 0.05 and rmse_gain >= 0.05:
            return True, "Both error measures improve by at least 5% versus baseline."
        if (
            model_r2 is not None
            and r2_gain is not None
            and model_r2 >= 0.10
            and r2_gain >= 0.05
        ):
            return True, "Held-out R² clears a practical threshold and improves materially."
        return False, "No material improvement over the simple baseline was demonstrated."

    model_accuracy = numeric_value(metrics.get("accuracy"))
    baseline_accuracy = numeric_value(baseline_metrics.get("accuracy"))
    model_f1 = numeric_value(metrics.get("f1"))
    baseline_f1 = numeric_value(baseline_metrics.get("f1"))

    if (
        model_accuracy is not None
        and baseline_accuracy is not None
        and model_f1 is not None
        and baseline_f1 is not None
        and model_accuracy - baseline_accuracy >= 0.05
        and model_f1 - baseline_f1 >= 0.05
    ):
        return True, "Accuracy and weighted F1 both improve by at least five points."
    return False, "No material improvement over the simple baseline was demonstrated."


def render_prediction_result(prediction: Mapping[str, Any]) -> None:
    """Render the serializable result returned by the predictive pipeline."""

    task = str(prediction.get("task", ""))
    metrics = prediction.get("metrics", {})
    baseline_metrics = prediction.get("baseline_metrics", {})
    quality = prediction.get("model_quality", {})

    if not isinstance(metrics, Mapping):
        metrics = {}
    if not isinstance(baseline_metrics, Mapping):
        baseline_metrics = {}
    if not isinstance(quality, Mapping):
        quality = {}

    render_section(
        "Model result",
        "Held-out validation",
        (
            "Performance is compared with a simple baseline, rather than "
            "reported only on the data used to fit the model."
        ),
    )

    if task == "regression":
        metric_columns = st.columns(3)
        regression_metrics = (
            ("R²", "r2", 3),
            ("MAE", "mae", 2),
            ("RMSE", "rmse", 2),
        )
        for container, (label, key, decimals) in zip(
            metric_columns, regression_metrics
        ):
            with container:
                render_metric_card(
                    label,
                    format_number(metrics.get(key), decimals),
                    f"Baseline: {format_number(baseline_metrics.get(key), decimals)}",
                )
        performance = pd.DataFrame(
            {
                "Model": {
                    "R²": metrics.get("r2"),
                    "MAE": metrics.get("mae"),
                    "RMSE": metrics.get("rmse"),
                },
                "Baseline": {
                    "R²": baseline_metrics.get("r2"),
                    "MAE": baseline_metrics.get("mae"),
                    "RMSE": baseline_metrics.get("rmse"),
                },
            }
        )
    else:
        metric_columns = st.columns(4)
        classification_metrics = (
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1", "f1"),
        )
        for container, (label, key) in zip(metric_columns, classification_metrics):
            with container:
                render_metric_card(
                    label,
                    format_percent(metrics.get(key)),
                    f"Baseline: {format_percent(baseline_metrics.get(key))}",
                )
        performance = pd.DataFrame(
            {
                "Model": {
                    "Accuracy": metrics.get("accuracy"),
                    "Precision": metrics.get("precision"),
                    "Recall": metrics.get("recall"),
                    "F1": metrics.get("f1"),
                },
                "Baseline": {
                    "Accuracy": baseline_metrics.get("accuracy"),
                    "Precision": baseline_metrics.get("precision"),
                    "Recall": baseline_metrics.get("recall"),
                    "F1": baseline_metrics.get("f1"),
                },
            }
        )

    with st.expander("Compare model and baseline", expanded=False):
        st.dataframe(performance, width="stretch")

    meaningful, quality_copy = meaningful_model_improvement(
        task, metrics, baseline_metrics
    )
    model_is_better = bool(quality.get("model_is_better", False))

    if meaningful:
        render_notice(
            "Model quality",
            "Meaningfully better than baseline",
            quality_copy,
            tone="success",
        )
    elif model_is_better:
        render_notice(
            "Model quality",
            "Marginal improvement — needs review",
            (
                "The model edges past the baseline numerically, but the gain is "
                "not yet large enough to treat as dependable business value. "
                + quality_copy
            ),
            tone="warning",
        )
    else:
        render_notice(
            "Model quality",
            "Needs review",
            (
                "The model did not consistently outperform the baseline on "
                "held-out data. Treat predictions as exploratory. " + quality_copy
            ),
            tone="warning",
        )

    source_rows = int(prediction.get("source_rows", 0) or 0)
    modeling_rows = int(prediction.get("modeling_rows", source_rows) or 0)
    train_rows = int(prediction.get("train_rows", 0) or 0)
    test_rows = int(prediction.get("test_rows", 0) or 0)
    sampled = bool(prediction.get("was_sampled", False))

    render_metric_row(
        [
            (
                "Estimator",
                str(prediction.get("model_name", "Unknown")),
                "Bounded pipeline choice",
                True,
            ),
            ("Training rows", f"{train_rows:,}", "Used to fit"),
            ("Validation rows", f"{test_rows:,}", "Held out"),
            (
                "Modeling sample",
                f"{modeling_rows:,}",
                (
                    f"from {source_rows:,} source rows"
                    if sampled
                    else "full eligible dataset"
                ),
            ),
        ]
    )

    features = prediction.get("feature_columns", [])
    excluded = prediction.get("excluded_features", [])
    feature_one, feature_two = st.columns(2)
    with feature_one:
        render_notice(
            "Features used",
            f"{len(features) if isinstance(features, list) else 0} eligible features",
            html_list(features, "No feature names returned."),
        )
    with feature_two:
        render_notice(
            "Excluded features",
            f"{len(excluded) if isinstance(excluded, list) else 0} excluded",
            html_list(
                excluded,
                "No columns were excluded by the model's feature rules.",
            ),
            tone="warning" if excluded else "accent",
        )

    actual_values = prediction.get("actual_values", [])
    predicted_values = prediction.get("predictions", [])
    if not isinstance(actual_values, list) or not isinstance(predicted_values, list):
        return

    display_length = min(
        len(actual_values),
        len(predicted_values),
        PREDICTION_DISPLAY_LIMIT,
    )
    if display_length < 2:
        st.info("There are not enough held-out observations to draw validation.")
        return

    validation = pd.DataFrame(
        {
            "Actual": actual_values[:display_length],
            "Predicted": predicted_values[:display_length],
        }
    )

    render_section(
        "Validation sample",
        "Actual versus predicted",
        "The view is capped to keep the dashboard fast and legible.",
    )

    if len(actual_values) > PREDICTION_DISPLAY_LIMIT:
        st.caption(
            f"Showing the first {PREDICTION_DISPLAY_LIMIT:,} of "
            f"{len(actual_values):,} held-out predictions."
        )

    if task == "regression":
        st.line_chart(validation, width="stretch")
    else:
        st.caption(
            "Classification outcomes are shown as a held-out confusion-style count."
        )
        matrix = pd.crosstab(
            validation["Actual"],
            validation["Predicted"],
            rownames=["Actual"],
            colnames=["Predicted"],
        )
        st.dataframe(matrix, width="stretch")

    with st.expander("Inspect validation records", expanded=False):
        st.dataframe(validation, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def view_overview(
    profile: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    file_name: str | None,
) -> None:
    """Render the landing view: the pitch, the current state, the next step."""

    render_html(
        """
        <div class="hero">
            <div class="eyebrow">AI-powered business analytics copilot</div>

            <h1>
                Turn business data<br>
                into decisions.
            </h1>

            <p>
                Upload your dataset and move through one evidence-driven workflow:
                understand what happened, discover what may happen next, and identify
                the actions your business should consider.
            </p>

            <div class="journey">
                <span>01 · Understand</span>
                <span>02 · Discover</span>
                <span>03 · Predict</span>
                <span>04 · Decide</span>
            </div>
        </div>
        """
    )

    if profile is None:
        render_notice(
            "Start here",
            "Upload a CSV in the sidebar",
            (
                "The workspace profiles the file first, then runs the "
                "deterministic analysis to generate insights, decisions, and "
                "auditable execution details. Dock items unlock as each stage "
                "becomes available."
            ),
        )
        render_metric_row(
            [
                (
                    "Understand",
                    "What happened?",
                    "Patterns, trends, and business evidence",
                ),
                (
                    "Predict",
                    "What's next?",
                    "Validated forecasts and outcome signals",
                ),
                (
                    "Decide",
                    "What should we do?",
                    "Evidence-backed business recommendations",
                ),
            ]
        )
        return

    render_metric_row(
        [
            (
                "Active dataset",
                str(file_name or "Uploaded CSV"),
                "Scoped to this file",
                True,
            ),
            ("Rows", f"{int(profile['rows']):,}", "Parsed from the CSV"),
            ("Columns", f"{int(profile['columns']):,}", "Fields available"),
            (
                "Missing values",
                f"{float(profile['missing_rate']):.1f}%",
                f"{int(profile['missing_cells']):,} cells",
            ),
        ]
    )

    if result is None:
        render_notice(
            "Next step",
            "Run the deterministic analysis",
            (
                "The file is profiled and the Data view is open. Run the "
                "analysis to unlock the descriptive, prescriptive, and "
                "technical views."
            ),
        )
        st.button(
            "Run business analysis",
            type="primary",
            width="stretch",
            on_click=request_analysis,
            key="overview_run_analysis",
        )
        return

    failed = result.get("failed_executions", [])
    completed = result.get("completed_executions", [])
    skipped = result.get("skipped_executions", [])

    if isinstance(failed, list) and failed:
        render_notice(
            "Workspace status",
            "Your business analysis is ready",
            (
                f"{len(failed):,} execution item(s) could not be completed. "
                "See the Decisions view for execution detail."
            ),
            tone="warning",
        )
    else:
        render_notice(
            "Pipeline status",
            "Your business analysis is ready",
            "The analytical pipeline has transformed the uploaded dataset into business findings, predictive signals, and decision support.",
            tone="success",
        )

        columns = st.columns(4)

    with columns[0]:
        metric_card(
            "Key findings",
            f"{int(result.get('insight_count', 0) or 0):,}",
            "Prioritized business signals",
        )

    with columns[1]:
        metric_card(
            "Recommended actions",
            f"{int(result.get('decision_count', 0) or 0):,}",
            "Evidence-backed decisions",
        )

    with columns[2]:
        metric_card(
            "Analyses completed",
            f"{len(completed) if isinstance(completed, list) else 0:,}",
            "Validated analytical checks",
        )

    with columns[3]:
        metric_card(
            "Items to review",
            f"{len(skipped) if isinstance(skipped, list) else 0:,}",
            "Insufficient evidence or support",
        )

    insights = result.get("insights", [])
    if isinstance(insights, list) and insights:
        headline = next(
            (item for item in insights if isinstance(item, Mapping)), None
        )
        if headline is not None:
            render_notice(
                "Top finding",
                str(headline.get("title", "Untitled insight")),
                str(headline.get("finding", "No explanation supplied.")),
            )


def view_data(
    dataframe: pd.DataFrame,
    profile: Mapping[str, Any],
    file_name: str,
) -> None:
    """Render the dataset structure and quality profile."""

    render_section(
        "01 · Data",
        "Dataset overview",
        (
            "A concise quality and structure profile before analytical "
            "interpretation begins."
        ),
    )

    file_col, source_col = st.columns([3, 1])
    with file_col:
        render_notice(
            "Active dataset",
            file_name,
            "CSV upload · results are scoped to this exact file.",
        )
    with source_col:
        render_metric_card(
            "Memory footprint",
            f"{float(profile['memory_mb']):.1f} MB",
            "In-memory DataFrame estimate",
        )

    render_metric_row(
        [
            ("Rows", f"{int(profile['rows']):,}", "Data rows parsed"),
            ("Columns", f"{int(profile['columns']):,}", "Fields available"),
            (
                "Missing values",
                f"{float(profile['missing_rate']):.1f}%",
                f"{int(profile['missing_cells']):,} cells",
            ),
            (
                "Duplicate rows",
                f"{int(profile['duplicate_rows']):,}",
                "Exact row duplicates",
            ),
        ]
    )

    quality_one, quality_two = st.columns(2)
    with quality_one:
        render_notice(
            "Field mix",
            (
                f"{len(profile['numeric_columns']):,} measures · "
                f"{len(profile['categorical_columns']):,} dimensions"
            ),
            (
                f"{len(profile['datetime_columns']):,} datetime fields · "
                f"{len(profile['identifier_columns']):,} identifier fields"
            ),
        )
    with quality_two:
        clean = (
            int(profile["missing_cells"]) == 0
            and int(profile["duplicate_rows"]) == 0
        )
        render_notice(
            "Quality checkpoint",
            (
                "Dataset looks complete"
                if clean
                else "Review data quality before decisions"
            ),
            (
                "No missing values or duplicate rows were found."
                if clean
                else (
                    f"{int(profile['missing_cells']):,} missing cells and "
                    f"{int(profile['duplicate_rows']):,} duplicate rows were found."
                )
            ),
            tone="success" if clean else "warning",
        )

    with st.expander("Preview data", expanded=True):
        preview = dataframe.head(PREVIEW_ROW_LIMIT).copy()

    # Streamlit/PyArrow can fail when a column contains mixed
    # Python types such as integers + strings. Convert only the
    # displayed preview to a consistent representation.
    for column in preview.columns:
        if preview[column].dtype == "object":
            preview[column] = preview[column].astype("string")

    st.dataframe(
        preview,
        width="stretch",
        height=360,
        hide_index=True,
    )

    with st.expander("Column quality", expanded=False):
        quality_table = pd.DataFrame(profile["quality_rows"]).sort_values(
            ["Missing %", "Unique"],
            ascending=[False, False],
        )
        st.dataframe(quality_table, width="stretch", hide_index=True)


def render_group_comparisons(analytics: Any) -> None:
    """Show up to two concrete descriptive comparisons when supplied."""

    if not isinstance(analytics, Mapping):
        return

    comparisons = analytics.get("group_comparisons", [])
    if not isinstance(comparisons, list) or not comparisons:
        return

    usable = [item for item in comparisons[:2] if isinstance(item, Mapping)]
    if not usable:
        return

    columns = st.columns(len(usable))
    for container, comparison in zip(columns, usable):
        metric = str(comparison.get("metric", "Measure"))
        category = str(comparison.get("category", "Group"))
        high_group = comparison.get("highest_group")
        low_group = comparison.get("lowest_group")
        high_value = comparison.get("highest_value")
        low_value = comparison.get("lowest_value")

        with container:
            render_html(
                f"""
                <div class="panel">
                    <div class="panel-label">Group comparison</div>
                    <div class="notice-title">
                        {safe_text(metric)} by {safe_text(category)}
                    </div>
                    <p class="panel-note">
                        Highest: {safe_text(high_group)} · {format_number(high_value)}
                        <br>
                        Lowest: {safe_text(low_group)} · {format_number(low_value)}
                    </p>
                </div>
                """
            )

            if high_group is not None and low_group is not None:
                chart = pd.DataFrame(
                    {
                        category: [str(high_group), str(low_group)],
                        metric: [high_value, low_value],
                    }
                ).set_index(category)
                try:
                    st.bar_chart(chart, width="stretch")
                except (TypeError, ValueError):
                    pass


def render_insights(insights: Any) -> None:
    """Render pipeline insights defensively."""

    if not isinstance(insights, list) or not insights:
        st.info("No significant insights were generated for this dataset.")
        return

    for index, insight in enumerate(insights, start=1):
        if not isinstance(insight, Mapping):
            continue

        priority = safe_text(insight.get("priority", "medium")).upper()
        insight_type = safe_text(insight.get("type", "general"))
        title = safe_text(insight.get("title", "Untitled insight"))
        finding = safe_text(insight.get("finding", "No explanation supplied."))

        metadata = []
        if insight.get("metric"):
            metadata.append(f"Metric: {safe_text(insight['metric'])}")
        if insight.get("category"):
            metadata.append(f"Dimension: {safe_text(insight['category'])}")
        metadata_markup = (
            f'<p class="panel-note">{" · ".join(metadata)}</p>'
            if metadata
            else ""
        )

        render_html(
            f"""
            <article class="insight">
                <div class="insight-meta">{priority} · {insight_type}</div>
                <h3>{index}. {title}</h3>
                <p>{finding}</p>
                {metadata_markup}
            </article>
            """
        )


def view_descriptive(result: Mapping[str, Any]) -> None:
    """Render the descriptive evidence produced by the pipeline."""

    render_section(
        "02 · Descriptive",
        "What happened?",
        "Key patterns and comparisons discovered by the analytics pipeline.",
    )
    render_group_comparisons(result.get("analytics", {}))
    render_insights(result.get("insights", []))


def view_predictive(dataframe: pd.DataFrame, signature: str) -> None:
    """Render the explicitly triggered, bounded predictive workflow."""

    render_section(
        "03 · What's Next?",
        "What is the data telling us about the future?",
        (
            "Choose a business outcome to predict. The model is trained only when "
            "you explicitly request it, keeping the workspace responsive and "
            "preventing unnecessary retraining."
        ),
    )

    try:
        candidates = detect_prediction_targets(dataframe)
    except (TypeError, ValueError) as exc:
        render_notice(
            "Predictive readiness",
            "Target detection is unavailable",
            str(exc),
            tone="danger",
        )
        return

    regression_candidates = candidates.get("regression_candidates", [])
    classification_candidates = candidates.get("classification_candidates", [])
    targets = list(
        dict.fromkeys([*regression_candidates, *classification_candidates])
    )

    if not targets:
        render_notice(
            "Predictive readiness",
            "No suitable prediction target detected",
            (
                "Add a numeric outcome for regression or a categorical outcome "
                "for classification, then upload the file again."
            ),
            tone="warning",
        )
        return

    target_tasks: dict[str, str] = {}
    for target in targets:
        try:
            target_tasks[target] = detect_prediction_task(dataframe, target)
        except (TypeError, ValueError):
            target_tasks[target] = "unavailable"

    selection, readiness_column = st.columns([2.2, 1])
    with selection:
        selected_target = st.selectbox(
            "Target to predict",
            options=targets,
            format_func=lambda name: (
                f"{name} · {target_tasks.get(name, 'unavailable').title()}"
            ),
            key=f"prediction_target_{signature[:12]}",
        )

    readiness = target_readiness(dataframe, selected_target)
    with readiness_column:
        render_notice(
            "Predictive readiness",
            "Ready to model" if readiness.get("ready") else "Not ready",
            (
                f"{readiness.get('usable_target_rows', 0):,} usable target values"
                if readiness.get("ready")
                else str(readiness.get("reason", "Choose another target."))
            ),
            tone="success" if readiness.get("ready") else "warning",
        )

    candidate_one, candidate_two = st.columns(2)
    with candidate_one:
        render_notice(
            "Regression candidates",
            f"{len(regression_candidates):,} numeric targets",
            html_list(regression_candidates, "None detected."),
        )
    with candidate_two:
        render_notice(
            "Classification candidates",
            f"{len(classification_candidates):,} categorical targets",
            html_list(classification_candidates, "None detected."),
        )

    st.caption(
        "The predictive pipeline safeguards large datasets by using a bounded "
        "modeling sample. This screen does not prepare features or start a "
        "model until you explicitly request it."
    )

    model_key = prediction_key(signature, selected_target)
    existing_result = st.session_state.predictive_results.get(model_key)

    run_model = st.button(
        (
            "Re-run bounded model"
            if existing_result is not None
            else "Train bounded model"
        ),
        type="primary",
        width="stretch",
        disabled=(
            not bool(readiness.get("ready"))
            or bool(st.session_state.prediction_busy)
        ),
        key=f"run_model_{signature[:12]}_{selected_target}",
    )

    if run_model:
        st.session_state.prediction_busy = True
        try:
            with st.status("Preparing a bounded model run…", expanded=True) as status:
                st.write("Checking target and feature eligibility.")
                st.write("Training and evaluating on held-out data.")
                prediction = run_prediction(dataframe, selected_target)
                status.update(
                    label="Model evaluation complete",
                    state="complete",
                    expanded=False,
                )

            updated_results = dict(st.session_state.predictive_results)
            updated_results[model_key] = prediction
            st.session_state.predictive_results = updated_results

            updated_errors = dict(st.session_state.prediction_errors)
            updated_errors.pop(model_key, None)
            st.session_state.prediction_errors = updated_errors
            existing_result = prediction

        except (TypeError, ValueError, KeyError) as exc:
            updated_errors = dict(st.session_state.prediction_errors)
            updated_errors[model_key] = str(exc)
            st.session_state.prediction_errors = updated_errors
            existing_result = None
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            updated_errors = dict(st.session_state.prediction_errors)
            updated_errors[model_key] = (
                f"The predictive workflow stopped unexpectedly. Details: {exc}"
            )
            st.session_state.prediction_errors = updated_errors
            existing_result = None
        finally:
            st.session_state.prediction_busy = False

    error_message = st.session_state.prediction_errors.get(model_key)
    if error_message:
        render_notice(
            "Prediction run",
            "Model evaluation could not be completed",
            error_message,
            tone="danger",
        )

    if existing_result is not None:
        render_prediction_result(existing_result)
    elif readiness.get("ready"):
        render_notice(
            "Next step",
            "Train one baseline model",
            (
                "Choose the target above and press Train bounded model. The "
                "result stays available for this dataset and target without "
                "retraining on later reruns."
            ),
        )


def view_decisions(result: Mapping[str, Any]) -> None:
    """Render decisions followed by auditable execution detail."""

    render_section(
        "04 · Decision Support",
        "What should the business do next?",
        (
            "Recommendations are generated from the evidence discovered in the "
            "analysis pipeline. Each recommendation can be traced back to the "
            "underlying analytical evidence."
        ),
    )

    decisions = result.get("decisions", [])
    if not isinstance(decisions, list) or not decisions:
        st.info("No business decisions were generated from the current findings.")
    else:
        for index, decision in enumerate(decisions, start=1):
            if not isinstance(decision, Mapping):
                continue

            render_html(
                f"""
                <article class="decision">
                    <div class="decision-index">Decision {index}</div>
                    <h3>
                        {safe_text(decision.get("decision", "No decision text available."))}
                    </h3>
                </article>
                """
            )

            detail_one, detail_two, detail_three = st.columns(3)
            with detail_one:
                st.caption("Recommended analysis")
                st.code(str(decision.get("recommended_analysis", "Not specified")))
            with detail_two:
                st.caption("Confidence")
                st.write(str(decision.get("confidence", "Not specified")).title())
            with detail_three:
                count = sum(
                    len(value)
                    for value in (
                        decision.get("evidence", []),
                        decision.get("metrics", []),
                        decision.get("dimensions", []),
                    )
                    if isinstance(value, list)
                )
                st.caption("Evidence references")
                st.write(count)

            limitations = decision.get("limitations", [])
            if isinstance(limitations, list) and limitations:
                with st.expander(f"Decision {index} limitations"):
                    for limitation in limitations:
                        st.write(f"• {limitation}")

    render_section(
        "Analytical execution",
        "How recommendations were tested",
        (
            "Each decision is mapped to an execution plan where the available "
            "data supports a safe deterministic analysis."
        ),
    )

    executions = result.get("execution_results", [])
    if not isinstance(executions, list) or not executions:
        st.info("There are no execution details for this analysis.")
        return

    for index, execution in enumerate(executions, start=1):
        if not isinstance(execution, Mapping):
            continue

        plan = execution.get("plan")
        decision = execution.get("decision")
        if isinstance(plan, Mapping):
            analysis_type = plan.get("analysis_type", "Unknown")
        elif isinstance(decision, Mapping):
            analysis_type = decision.get("recommended_analysis", "Not executable")
        else:
            analysis_type = "Not executable"

        render_html(
            f"""
            <div class="panel" style="margin: 0.7rem 0;">
                <div class="panel-label">Execution {index}</div>
                <div class="notice-title">{safe_text(analysis_type)}</div>
                <div style="margin-top: 0.45rem;">
                    {status_pill(execution.get("status"))}
                </div>
            </div>
            """
        )

        with st.expander(f"Inspect execution {index}"):
            if isinstance(decision, Mapping):
                st.markdown("**Decision**")
                st.write(decision.get("decision", "No decision detail available."))
            if isinstance(plan, Mapping):
                st.markdown("**Execution plan**")
                st.json(dict(plan), expanded=False)
            if isinstance(execution.get("result"), Mapping):
                st.markdown("**Result**")
                st.json(dict(execution["result"]), expanded=False)
            if execution.get("error"):
                st.warning(str(execution["error"]))


def view_technical(
    result: Mapping[str, Any],
    profile: Mapping[str, Any],
    signature: str,
) -> None:
    """Expose diagnostics without crowding the business-facing views."""

    render_section(
        "05 · Technical",
        "Diagnostics and traceability",
        (
            "Inspect the deterministic context, data profile, and pipeline "
            "outputs behind this workspace."
        ),
    )

    diagnostics = {
        "dataset_fingerprint": signature[:16],
        "rows": profile.get("rows"),
        "columns": profile.get("columns"),
        "memory_mb": round(float(profile.get("memory_mb", 0.0)), 2),
        "pipeline_status": result.get("status", "unknown"),
        "insight_count": result.get("insight_count", 0),
        "decision_count": result.get("decision_count", 0),
        "execution_count": result.get("execution_count", 0),
    }

    with st.expander("Run diagnostics", expanded=True):
        st.json(diagnostics, expanded=False)
    with st.expander("Detected semantic roles", expanded=False):
        st.json(result.get("semantic_roles", {}), expanded=False)
    with st.expander("Analysis plan", expanded=False):
        st.json(result.get("analysis_plan", {}), expanded=False)
    with st.expander("Business questions", expanded=False):
        questions = result.get("business_questions", [])
        if isinstance(questions, list) and questions:
            for question in questions:
                if isinstance(question, Mapping):
                    text = (
                        question.get("question")
                        or question.get("business_question")
                        or str(question)
                    )
                else:
                    text = str(question)
                st.write(f"• {text}")
        else:
            st.info("No business questions are available.")
    with st.expander("Raw analytics", expanded=False):
        st.json(result.get("analytics", {}), expanded=False)


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


def status_chip_markup(
    file_name: str | None,
    profile: Mapping[str, Any] | None,
    has_analysis: bool,
) -> str:
    """Build the compact workspace status chip shown in the top bar."""

    if profile is None:
        return "No dataset loaded · <b>upload a CSV or XLSX file</b>"

    shape = f"{int(profile['rows']):,} × {int(profile['columns']):,}"
    state = "analysis ready" if has_analysis else "awaiting analysis"
    return f"{safe_text(file_name, 'Uploaded dataset')} · <b>{shape}</b> · {state}"


def main() -> None:
    """Compose the shell, resolve the active view, and render it."""

    initialise_session_state()
    render_styles()
    render_theme_bridge()

    uploaded_file, options = render_sidebar()

    dataframe: pd.DataFrame | None = None
    profile: dict[str, Any] | None = None
    signature: str | None = None
    load_error: str | None = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        signature = dataset_signature(file_bytes)

        if st.session_state.dataset_signature != signature:
            clear_dataset_results()
            st.session_state.dataset_signature = signature

        try:
            dataframe, profile = load_dataset_bundle(
                file_bytes,
                str(getattr(uploaded_file, "name", "")),
            )
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            load_error = str(exc)
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            load_error = f"Unexpected import issue: {exc}"

    has_dataset = dataframe is not None and profile is not None
    has_analysis = (
        st.session_state.analysis_result is not None
        and st.session_state.analysis_signature == signature
    )

    # The top bar keeps its place at the head of the page but is filled in
    # last, so its status chip reflects the run that happens below it.
    topbar_slot = st.empty()

    if load_error is not None:
        render_notice(
            "Dataset import",
            "The uploaded file could not be read",
            load_error,
            tone="danger",
        )

    # A queued run happens before the dock is drawn, so the dock reflects the
    # views that the finished analysis unlocks.
    if st.session_state.analysis_requested:
        st.session_state.analysis_requested = False
        if has_dataset and dataframe is not None and signature is not None:
            execute_analysis(dataframe, signature, options)
            has_analysis = st.session_state.analysis_result is not None

    topbar_slot.html(
        topbar_markup(
            status_chip_markup(
                getattr(uploaded_file, "name", None),
                profile,
                has_analysis,
            )
        )
    )

    if st.session_state.analysis_error:
        render_notice(
            "Business analysis",
            "Analysis could not be completed",
            str(st.session_state.analysis_error),
            tone="danger",
        )

    if (
        has_analysis
        and st.session_state.analysis_options is not None
        and st.session_state.analysis_options != options
    ):
        st.info("Analysis controls changed. Run the analysis again to apply them.")

    active_view = resolve_view(has_dataset, has_analysis)
    render_dock(active_view, has_dataset, has_analysis)

    if st.session_state.scroll_pending:
        st.session_state.scroll_pending = False
        scroll_main_to_top()

    result = st.session_state.analysis_result if has_analysis else None

    if active_view == "overview":
        view_overview(profile, result, getattr(uploaded_file, "name", None))
    elif active_view == "data" and dataframe is not None and profile is not None:
        view_data(dataframe, profile, str(getattr(uploaded_file, "name", "Dataset")))
    elif active_view == "descriptive" and result is not None:
        view_descriptive(result)
    elif active_view == "predictive" and dataframe is not None and signature:
        view_predictive(dataframe, signature)
    elif active_view == "decisions" and result is not None:
        view_decisions(result)
    elif (
        active_view == "technical"
        and result is not None
        and profile is not None
        and signature
    ):
        view_technical(result, profile, signature)

    render_footer()


main()
