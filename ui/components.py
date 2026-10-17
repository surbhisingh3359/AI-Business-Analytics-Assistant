import html

import streamlit as st


def _safe_text(value: object) -> str:
    """Escape text before rendering it inside HTML."""
    return html.escape(str(value))


def section_header(
    eyebrow: str,
    title: str,
    description: str = "",
) -> None:
    """Render a consistent section header across the dashboard."""

    safe_eyebrow = _safe_text(eyebrow)
    safe_title = _safe_text(title)
    safe_description = _safe_text(description)

    description_html = (
        f"<p>{safe_description}</p>" if description else ""
    )

    st.markdown(
        f"""
        <div class="section-header">
            <div class="eyebrow">{safe_eyebrow}</div>
            <h2>{safe_title}</h2>
            {description_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: str,
    description: str = "",
) -> None:
    """Render a reusable KPI card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-description">{description}</div>' if description else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(
    label: str,
    status: str = "neutral",
) -> None:
    """Render a small status badge."""

    allowed_statuses = {"success", "warning", "danger", "neutral"}

    if status not in allowed_statuses:
        status = "neutral"

    st.markdown(
        f"""
        <span class="status-badge status-{status}">
            {label}
        </span>
        """,
        unsafe_allow_html=True,
    )
