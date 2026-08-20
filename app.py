import os
import tempfile

import streamlit as st
import pandas as pd

from src.preprocessing import preprocess_data
from src.dataset_classifier import detect_semantic_roles
from src.analysis_planner import build_analysis_plan
from src.universal_analytics import analyze_dataset
from src.ai_analysis import analyze_business_data


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Business Analytics Assistant",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================

st.title("📊 AI Business Analytics Assistant")

st.write(
    """
    Upload any structured CSV dataset and let the application
    automatically detect its structure, determine suitable
    analyses, generate insights, and provide AI-powered
    business explanations.
    """
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
)


if uploaded_file is None:

    st.info(
        "Upload a CSV file to begin automatic dataset analysis."
    )

    st.stop()


# =========================================================
# SAVE UPLOADED FILE
# =========================================================

st.success(
    f"Uploaded: {uploaded_file.name}"
)

with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".csv",
) as temp_file:

    temp_file.write(
        uploaded_file.getbuffer()
    )

    temp_file_path = temp_file.name


try:

    # =====================================================
    # DATA PREPROCESSING
    # =====================================================

    with st.spinner("Cleaning and preparing your dataset..."):

        preprocessing_result = preprocess_data(
            temp_file_path
        )

    cleaned_df = preprocessing_result["data"]

    quality_report = preprocessing_result[
        "quality_report"
    ]


    if cleaned_df.empty:

        st.error(
            "The uploaded dataset contains no usable rows."
        )

        st.stop()


    # =====================================================
    # DATASET OVERVIEW
    # =====================================================

    st.header("📋 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            f"{len(cleaned_df):,}",
        )

    with col2:

        st.metric(
            "Columns",
            f"{len(cleaned_df.columns):,}",
        )

    with col3:

        st.metric(
            "Missing Values",
            f"{int(cleaned_df.isna().sum().sum()):,}",
        )

    with col4:

        st.metric(
            "Duplicate Rows",
            f"{int(cleaned_df.duplicated().sum()):,}",
        )


    # =====================================================
    # DATA PREVIEW
    # =====================================================

    with st.expander(
        "🔎 View Dataset Preview",
        expanded=True,
    ):

        st.dataframe(
            cleaned_df.head(100),
            use_container_width=True,
        )


    # =====================================================
    # SEMANTIC CLASSIFICATION
    # =====================================================

    st.header("🧠 Dataset Understanding")

    with st.spinner(
        "Automatically identifying column meanings..."
    ):

        semantic_roles = detect_semantic_roles(
            cleaned_df
        )


    role_df = pd.DataFrame(
        {
            "Column": list(
                semantic_roles.keys()
            ),
            "Detected Role": list(
                semantic_roles.values()
            ),
        }
    )

    st.dataframe(
        role_df,
        use_container_width=True,
        hide_index=True,
    )


    # =====================================================
    # ANALYSIS PLAN
    # =====================================================

    with st.spinner(
        "Building an analysis plan for this dataset..."
    ):

        analysis_plan = build_analysis_plan(
            cleaned_df,
            semantic_roles,
        )


    st.header("📐 Automatic Analysis Plan")

    analyses = analysis_plan.get(
        "analyses",
        [],
    )

    if analyses:

        st.write(
            "The application automatically selected "
            "the following analyses:"
        )

        for analysis in analyses:

            st.markdown(
                f"• `{analysis}`"
            )

    else:

        st.warning(
            "No specialized analyses were identified."
        )


    # =====================================================
    # UNIVERSAL ANALYTICS
    # =====================================================

    with st.spinner(
        "Running automatic dataset analysis..."
    ):

        analytics_result = analyze_dataset(
            cleaned_df,
            semantic_roles,
            analysis_plan,
        )


    # =====================================================
    # NUMERIC ANALYSIS
    # =====================================================

    numeric_summary = analytics_result.get(
        "numeric_summary",
        {},
    )

    if numeric_summary:

        st.header("🔢 Numeric Analysis")

        numeric_df = pd.DataFrame(
            numeric_summary
        ).T

        numeric_df.index.name = "Column"

        st.dataframe(
            numeric_df,
            use_container_width=True,
        )


    # =====================================================
    # CATEGORICAL ANALYSIS
    # =====================================================

    categorical_summary = analytics_result.get(
        "categorical_summary",
        {},
    )

    if categorical_summary:

        st.header("🏷️ Categorical Analysis")

        categorical_df = pd.DataFrame(
            categorical_summary
        ).T

        categorical_df.index.name = "Column"

        st.dataframe(
            categorical_df,
            use_container_width=True,
        )


    # =====================================================
    # DATETIME ANALYSIS
    # =====================================================

    datetime_summary = analytics_result.get(
        "datetime_summary",
        {},
    )

    if datetime_summary:

        st.header("📅 Time Analysis")

        datetime_df = pd.DataFrame(
            datetime_summary
        ).T

        datetime_df.index.name = "Column"

        st.dataframe(
            datetime_df,
            use_container_width=True,
        )


    # =====================================================
    # CORRELATION ANALYSIS
    # =====================================================

    correlations = analytics_result.get(
        "correlations",
        {},
    )

    if correlations:

        st.header("🔗 Correlation Analysis")

        correlation_df = pd.DataFrame(
            correlations
        )

        st.dataframe(
            correlation_df,
            use_container_width=True,
        )


    # =====================================================
    # GROUP COMPARISON
    # =====================================================

    group_comparisons = analytics_result.get(
        "group_comparisons",
        {},
    )

    if group_comparisons:

        st.header("📊 Group Comparisons")

        for category, metrics in group_comparisons.items():

            st.subheader(
                f"Analysis by {category}"
            )

            for metric, values in metrics.items():

                comparison_df = (
                    pd.Series(values)
                    .sort_values(
                        ascending=False
                    )
                    .to_frame(
                        name="Average"
                    )
                )

                comparison_df.index.name = category

                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                )


    # =====================================================
    # DATA QUALITY
    # =====================================================

    st.header("🧹 Data Quality")

    before = quality_report.get(
        "before",
        {},
    )

    after = quality_report.get(
        "after",
        {},
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows Before Cleaning",
            before.get(
                "rows",
                len(cleaned_df),
            ),
        )

    with col2:

        st.metric(
            "Rows After Cleaning",
            after.get(
                "rows",
                len(cleaned_df),
            ),
        )

    with col3:

        duplicates_removed = (
            before.get("duplicates", 0)
            - after.get("duplicates", 0)
        )

        st.metric(
            "Duplicates Removed",
            max(
                duplicates_removed,
                0,
            ),
        )


    # =====================================================
    # AI ANALYSIS
    # =====================================================

    st.header("🤖 AI Business Analysis")

    st.write(
        """
        The AI will interpret the automatically detected
        dataset structure and available analysis results.
        """
    )

    if st.button(
        "Generate AI Analysis",
        type="primary",
    ):

        with st.spinner(
            "Generating AI-powered analysis..."
        ):

            try:

                ai_result = analyze_business_data(
                    temp_file_path
                )

                analysis = ai_result.get(
                    "analysis",
                    "",
                )

                validation = ai_result.get(
                    "validation",
                    {},
                )

                if analysis:

                    st.markdown(
                        analysis
                    )

                else:

                    st.warning(
                        "The AI did not return an analysis."
                    )


                # =========================================
                # VALIDATION
                # =========================================

                st.divider()

                st.subheader(
                    "🛡️ AI Response Quality"
                )

                if validation.get(
                    "is_valid",
                    False,
                ):

                    st.success(
                        "✅ AI analysis passed validation."
                    )

                else:

                    st.warning(
                        "⚠️ AI analysis requires review."
                    )

                    warnings = validation.get(
                        "warnings",
                        [],
                    )

                    if warnings:

                        with st.expander(
                            "View validation warnings"
                        ):

                            for warning in warnings:

                                st.write(
                                    f"• {warning}"
                                )


                if validation.get(
                    "has_recommendations",
                    False,
                ):

                    st.info(
                        "💡 Actionable recommendations detected."
                    )

            except Exception as e:

                st.error(
                    f"AI analysis failed: {e}"
                )


except Exception as e:

    st.error(
        f"Unable to process the uploaded dataset: {e}"
    )


finally:

    if os.path.exists(
        temp_file_path
    ):

        os.remove(
            temp_file_path
        )