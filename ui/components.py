import streamlit as st


def section_header(
    eyebrow: str,
    title: str,
    description: str = "",
) -> None:
    """Render a consistent section header across the dashboard."""

    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-eyebrow">{eyebrow}</div>
            <h2>{title}</h2>
            {f'<p>{description}</p>' if description else ''}
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