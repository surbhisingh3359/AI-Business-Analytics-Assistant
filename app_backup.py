import io
from typing import Any

import pandas as pd
import streamlit as st

from src.business_analysis_pipeline import (
    run_business_analysis,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Business Analytics Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 650;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }

        .insight-card {
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 10px;
            margin-bottom: 0.8rem;
        }

        .small-label {
            color: #666;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 AI Business Analytics Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Upload a dataset and turn raw business data into "
    "insights, decisions, and executable analysis."
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Analysis Settings")

    max_insights = st.slider(
        "Maximum insights",
        min_value=1,
        max_value=10,
        value=5,
    )

    max_decisions = st.slider(
        "Maximum decisions",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.divider()

    st.markdown(
        """
        **Pipeline**

        Upload → Analyze → Insights → Decisions → Execution
        """
    )


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"],
    help="Upload a business dataset in CSV format.",
)


if uploaded_file is None:

    st.info(
        "Upload a CSV file to begin the analysis."
    )

    st.markdown(
        """
        ### What this application does

        **1. Understands your dataset**
        Detects semantic roles and dataset characteristics.

        **2. Performs universal analytics**
        Calculates summaries, relationships, group comparisons,
        trends, and other applicable analyses.

        **3. Generates insights**
        Converts analytical results into meaningful findings.

        **4. Ranks insights**
        Prioritizes findings according to impact and confidence.

        **5. Creates business decisions**
        Converts important findings into decision-oriented questions.

        **6. Executes recommended analysis**
        Turns decisions into concrete analytical operations.
        """
    )

    st.stop()


# =========================================================
# READ DATASET
# =========================================================

try:

    file_bytes = uploaded_file.getvalue()

    dataframe = pd.read_csv(
        io.BytesIO(file_bytes)
    )

except Exception as exc:

    st.error(
        f"Unable to read the uploaded CSV: {exc}"
    )

    st.stop()


# =========================================================
# PREVIEW
# =========================================================

st.markdown(
    '<div class="section-title">📄 Dataset Preview</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Rows",
        f"{len(dataframe):,}",
    )

with col2:
    st.metric(
        "Columns",
        f"{len(dataframe.columns):,}",
    )

with col3:
    st.metric(
        "Missing Cells",
        f"{int(dataframe.isna().sum().sum()):,}",
    )

with col4:
    st.metric(
        "Duplicate Rows",
        f"{int(dataframe.duplicated().sum()):,}",
    )

with st.expander(
    "View uploaded data",
    expanded=False,
):
    st.dataframe(
        dataframe,
        use_container_width=True,
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze_button = st.button(
    "🚀 Analyze Dataset",
    type="primary",
    use_container_width=True,
)


if analyze_button:

    with st.spinner(
        "Running business analysis..."
    ):

        try:

            result = run_business_analysis(
                dataframe,
                max_insights=max_insights,
                max_decisions=max_decisions,
            )

        except Exception as exc:

            st.error(
                f"Analysis failed: {exc}"
            )

            st.exception(
                exc
            )

            st.stop()

    st.session_state[
        "analysis_result"
    ] = result


# =========================================================
# DISPLAY RESULTS
# =========================================================

result = st.session_state.get(
    "analysis_result"
)


if result is None:
    st.info(
        "Click **Analyze Dataset** to generate the business analysis."
    )

    st.stop()


# =========================================================
# STATUS
# =========================================================

status = result.get(
    "status",
    "unknown",
)

if status == "completed":

    st.success(
        "Analysis completed successfully."
    )

elif status == "completed_with_errors":

    st.warning(
        "Analysis completed, but some recommended analyses failed."
    )

else:

    st.info(
        f"Analysis status: {status}"
    )


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">📌 Executive Overview</div>',
    unsafe_allow_html=True,
)

overview_col1, overview_col2, overview_col3, overview_col4 = (
    st.columns(4)
)

with overview_col1:

    st.metric(
        "Insights",
        result.get(
            "insight_count",
            0,
        ),
    )

with overview_col2:

    st.metric(
        "Decisions",
        result.get(
            "decision_count",
            0,
        ),
    )

with overview_col3:

    st.metric(
        "Executions",
        result.get(
            "execution_count",
            0,
        ),
    )

with overview_col4:

    st.metric(
        "Successful Executions",
        len(
            result.get(
                "completed_executions",
                [],
            )
        ),
    )


# =========================================================
# DATASET INFORMATION
# =========================================================

with st.expander(
    "🔎 Dataset Information",
    expanded=False,
):

    dataset_info = result.get(
        "dataset",
        {},
    )

    st.json(
        dataset_info
    )


# =========================================================
# SEMANTIC ROLES
# =========================================================

with st.expander(
    "🧠 Detected Semantic Roles",
    expanded=False,
):

    semantic_roles = result.get(
        "semantic_roles",
        {},
    )

    if semantic_roles:

        role_df = pd.DataFrame(
            [
                {
                    "Column": column,
                    "Role": role,
                }
                for column, role
                in semantic_roles.items()
            ]
        )

        st.dataframe(
            role_df,
            use_container_width=True,
        )

    else:

        st.info(
            "No semantic role information available."
        )


# =========================================================
# ANALYSIS PLAN
# =========================================================

with st.expander(
    "🗂️ Analysis Plan",
    expanded=False,
):

    analysis_plan = result.get(
        "analysis_plan",
        {},
    )

    st.json(
        analysis_plan
    )


# =========================================================
# KEY INSIGHTS
# =========================================================

st.markdown(
    '<div class="section-title">💡 Key Insights</div>',
    unsafe_allow_html=True,
)

insights = result.get(
    "insights",
    [],
)

if not insights:

    st.info(
        "No significant insights were generated."
    )

else:

    for index, insight in enumerate(
        insights,
        start=1,
    ):

        priority = str(
            insight.get(
                "priority",
                "medium",
            )
        ).upper()

        insight_type = str(
            insight.get(
                "type",
                "general",
            )
        )

        title = insight.get(
            "title",
            "Untitled Insight",
        )

        finding = insight.get(
            "finding",
            "",
        )

        with st.container(
            border=True
        ):

            st.markdown(
                f"### {index}. {title}"
            )

            meta_col1, meta_col2 = st.columns(2)

            with meta_col1:
                st.caption(
                    f"Priority: {priority}"
                )

            with meta_col2:
                st.caption(
                    f"Type: {insight_type}"
                )

            st.write(
                finding
            )

            metric = insight.get(
                "metric"
            )

            category = insight.get(
                "category"
            )

            if metric or category:

                metadata = []

                if metric:
                    metadata.append(
                        f"Metric: `{metric}`"
                    )

                if category:
                    metadata.append(
                        f"Dimension: `{category}`"
                    )

                st.caption(
                    " • ".join(metadata)
                )


# =========================================================
# BUSINESS DECISIONS
# =========================================================

st.markdown(
    '<div class="section-title">🎯 Business Decisions</div>',
    unsafe_allow_html=True,
)

decisions = result.get(
    "decisions",
    [],
)

if not decisions:

    st.info(
        "No business decisions were generated."
    )

else:

    for index, decision in enumerate(
        decisions,
        start=1,
    ):

        with st.container(
            border=True
        ):

            st.markdown(
                f"### Decision {index}"
            )

            st.write(
                decision.get(
                    "decision",
                    "",
                )
            )

            rec_col1, rec_col2 = st.columns(2)

            with rec_col1:

                st.caption(
                    "Recommended Analysis"
                )

                st.code(
                    decision.get(
                        "recommended_analysis",
                        "N/A",
                    )
                )

            with rec_col2:

                st.caption(
                    "Confidence"
                )

                st.write(
                    decision.get(
                        "confidence",
                        "N/A",
                    )
                )

            metrics = decision.get(
                "metrics",
                [],
            )

            dimensions = decision.get(
                "dimensions",
                [],
            )

            if metrics:

                st.caption(
                    "Metrics: "
                    + ", ".join(
                        str(item)
                        for item in metrics
                    )
                )

            if dimensions:

                st.caption(
                    "Dimensions: "
                    + ", ".join(
                        str(item)
                        for item in dimensions
                    )
                )

            limitations = decision.get(
                "limitations",
                [],
            )

            if limitations:

                with st.expander(
                    "Analysis limitations"
                ):

                    for limitation in limitations:

                        st.write(
                            f"- {limitation}"
                        )


# =========================================================
# EXECUTION RESULTS
# =========================================================

st.markdown(
    '<div class="section-title">⚙️ Executed Analysis</div>',
    unsafe_allow_html=True,
)

execution_results = result.get(
    "execution_results",
    [],
)

if not execution_results:

    st.info(
        "No execution results available."
    )

else:

    for index, execution in enumerate(
        execution_results,
        start=1,
    ):

        execution_status = execution.get(
            "status",
            "unknown",
        )

        if execution_status == "completed":

            st.success(
                f"Execution {index}: completed"
            )

        else:

            st.error(
                f"Execution {index}: failed"
            )

        decision = execution.get(
            "decision",
            {},
        )

        plan = execution.get(
            "plan",
            {},
        )

        execution_result = execution.get(
            "result",
        )

        with st.expander(
            f"Execution {index}: "
            f"{plan.get('analysis_type', 'Unknown')}"
        ):

            st.markdown(
                "**Decision**"
            )

            st.write(
                decision.get(
                    "decision",
                    "",
                )
            )

            st.markdown(
                "**Execution Plan**"
            )

            st.json(
                plan
            )

            if execution_result is not None:

                st.markdown(
                    "**Result**"
                )

                st.json(
                    execution_result
                )

            if execution.get(
                "error"
            ):

                st.error(
                    execution["error"]
                )


# =========================================================
# BUSINESS QUESTIONS
# =========================================================

business_questions = result.get(
    "business_questions",
    [],
)

if business_questions:

    st.markdown(
        '<div class="section-title">❓ Business Questions</div>',
        unsafe_allow_html=True,
    )

    for question in business_questions:

        if isinstance(
            question,
            dict,
        ):

            text = (
                question.get(
                    "question"
                )
                or question.get(
                    "business_question"
                )
                or str(question)
            )

        else:

            text = str(
                question
            )

        st.write(
            f"• {text}"
        )


# =========================================================
# RAW ANALYTICS
# =========================================================

with st.expander(
    "📊 Raw Analytics",
    expanded=False,
):

    analytics = result.get(
        "analytics",
        {},
    )

    st.json(
        analytics
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Business Analytics Assistant • "
    "Deterministic analytical core"
)