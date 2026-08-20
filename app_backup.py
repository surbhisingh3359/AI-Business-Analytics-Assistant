import os
import tempfile

import streamlit as st

from src.preprocessing import preprocess_data
from src.ai_context import build_ai_context
from src.ai_analysis import analyze_business_data
from src.visualizations import (
    revenue_by_category,
    revenue_by_product,
    quantity_by_product,
)


st.set_page_config(
    page_title="AI Business Analytics Assistant",
    page_icon="📊",
    layout="wide",
)


st.title("📊 AI Business Analytics Assistant")

st.write(
    "Upload a business CSV file and generate "
    "data-driven business insights using local AI."
)


uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
)


if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    # ---------------------------------------------------------
    # Save uploaded file temporarily
    # ---------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv",
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    try:

        # -----------------------------------------------------
        # Process uploaded CSV
        # -----------------------------------------------------

        result = preprocess_data(temp_file_path)

        cleaned_df = result["data"]
        quality_report = result["quality_report"]

        # -----------------------------------------------------
        # Build business context
        # -----------------------------------------------------

        context = build_ai_context(cleaned_df)

        summary = context["business_summary"]
        insights = context["business_insights"]

        # -----------------------------------------------------
        # Dataset Preview
        # -----------------------------------------------------

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            cleaned_df,
            use_container_width=True,
        )

        # -----------------------------------------------------
        # Business KPIs
        # -----------------------------------------------------

        st.subheader("📊 Business Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Revenue",
                f"₹{summary['total_revenue']:,.2f}",
            )

        with col2:
            st.metric(
                "Total Orders",
                f"{summary['total_orders']:,}",
            )

        with col3:
            st.metric(
                "Quantity Sold",
                f"{summary['total_quantity']:,}",
            )

        with col4:
            st.metric(
                "Average Order Value",
                f"₹{summary['average_order_value']:,.2f}",
            )

        # -----------------------------------------------------
        # Top Business Performers
        # -----------------------------------------------------

        st.subheader("🏆 Top Business Performers")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 🛒 Top Product")

            st.write(
                f"**{summary['top_product']['product']}**"
            )

            st.write(
                f"Revenue: ₹"
                f"{summary['top_product']['revenue']:,.2f}"
            )

        with col2:

            st.markdown("### 📦 Top Category")

            st.write(
                f"**{summary['top_category']['category']}**"
            )

            st.write(
                f"Revenue: ₹"
                f"{summary['top_category']['revenue']:,.2f}"
            )

        # -----------------------------------------------------
        # Business Insights
        # -----------------------------------------------------

        st.subheader("💡 Business Insights")

        top_category = insights["top_category"]
        top_product = insights["top_product"]

        st.write(
            f"**Top Category:** {top_category['category']}"
        )

        st.write(
            f"Category Revenue Contribution: "
            f"{top_category['contribution_percentage']:.2f}%"
        )

        st.write(
            f"**Top Product:** {top_product['product']}"
        )

        st.write(
            f"Product Revenue Concentration: "
            f"{top_product['concentration_percentage']:.2f}%"
        )

        risk_level = top_product["risk_level"]

        if risk_level == "high":

            st.error(
                f"⚠️ Revenue Concentration Risk: "
                f"{risk_level.upper()}"
            )

        elif risk_level == "medium":

            st.warning(
                f"⚠️ Revenue Concentration Risk: "
                f"{risk_level.upper()}"
            )

        else:

            st.success(
                f"Revenue Concentration Risk: "
                f"{risk_level.upper()}"
            )

        st.info(
            insights["product_concentration_insight"]
        )

        # -----------------------------------------------------
        # Business Visualizations
        # -----------------------------------------------------

        st.subheader("📈 Business Visualizations")

        category_revenue = revenue_by_category(
            cleaned_df
        )

        product_revenue = revenue_by_product(
            cleaned_df
        )

        product_quantity = quantity_by_product(
            cleaned_df
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Revenue by Category")

            st.bar_chart(
                category_revenue.set_index(
                    "category"
                )["revenue"]
            )

        with col2:

            st.markdown("#### Revenue by Product")

            st.bar_chart(
                product_revenue.set_index(
                    "product"
                )["revenue"]
            )

        st.markdown("#### Quantity Sold by Product")

        st.bar_chart(
            product_quantity.set_index(
                "product"
            )["quantity"]
        )

        # -----------------------------------------------------
        # AI Business Analysis
        # -----------------------------------------------------

        st.subheader("🤖 AI Business Analysis")

        if st.button(
            "Generate AI Business Analysis",
            type="primary",
        ):

            with st.spinner(
                "Analyzing your business data with local AI..."
            ):

                try:

                    ai_result = analyze_business_data(
                        temp_file_path
                    )

                    # ---------------------------------------------
                    # AI Analysis
                    # ---------------------------------------------

                    st.markdown(
                        ai_result["analysis"]
                    )

                    # ---------------------------------------------
                    # AI Validation
                    # ---------------------------------------------

                    validation = ai_result["validation"]

                    st.divider()

                    st.subheader(
                        "🛡️ AI Response Quality"
                    )

                    if validation["is_valid"]:

                        st.success(
                            "✅ AI analysis passed validation."
                        )

                    else:

                        st.warning(
                            "⚠️ AI analysis requires review."
                        )

                        with st.expander(
                            "View validation warnings"
                        ):

                            for warning in validation["warnings"]:

                                st.write(
                                    f"- {warning}"
                                )

                    if validation["has_recommendations"]:

                        st.info(
                            "💡 Actionable recommendations detected."
                        )

                except Exception as e:

                    st.error(
                        f"AI analysis failed: {e}"
                    )

        # -----------------------------------------------------
        # Data Quality
        # -----------------------------------------------------

        st.subheader("🧹 Data Quality")

        quality_col1, quality_col2, quality_col3 = st.columns(3)

        with quality_col1:

            st.metric(
                "Rows Before Cleaning",
                quality_report["before"]["rows"],
            )

        with quality_col2:

            st.metric(
                "Rows After Cleaning",
                quality_report["after"]["rows"],
            )

        with quality_col3:

            st.metric(
                "Duplicates Removed",
                quality_report["before"]["duplicates"]
                - quality_report["after"]["duplicates"],
            )

    except Exception as e:

        st.error(
            f"Unable to process the uploaded file: {e}"
        )

    finally:

        if os.path.exists(temp_file_path):

            os.remove(temp_file_path)