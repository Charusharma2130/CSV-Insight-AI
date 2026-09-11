import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# ACTUAL APP.PY BACKEND
# =========================================================

try:
    from app import (
        automatic_insights,
        correlation_analysis,
        outlier_analysis,
        data_quality_analysis,
        gemini_deep_analysis,
        generate_ai_analysis_report,
    )
    BACKEND_READY = True
    BACKEND_ERROR = ""
except Exception as e:
    BACKEND_READY = False
    BACKEND_ERROR = str(e)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    /* Main page */
    .stApp {
        background: #0e1117;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    /* Header */
    .hero {
        padding: 1.2rem 0 1.8rem 0;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        color: #a9b1bd;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* Section titles */
    .section-title { 
        font-size: 1.55rem;
        font-weight: 750;
        margin-top: 2rem;
        margin-bottom: 0.9rem;
    }

    .section-caption {
        color: #8f98a5;
        font-size: 0.92rem;
        margin-top: -0.55rem;
        margin-bottom: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: #171b24;
        border: 1px solid #292f3b;
        border-radius: 16px;
        padding: 1.15rem 1.2rem;
        min-height: 125px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.16);
    }

    .metric-label {
        color: #9da6b3;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.55rem;
    }

    .metric-value {
        color: #f3f5f7;
        font-size: 2rem;
        font-weight: 750;
        line-height: 1.1;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background: #151922;
        border: 1px dashed #3a4352;
        border-radius: 16px;
        padding: 0.35rem;
    }

    /* Success message */
    .upload-success {
        background: #123b2b;
        border: 1px solid #1d6b4a;
        color: #baf5d7;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        margin: 0.8rem 0 1.3rem 0;
        font-weight: 600;
    }

    /* Info cards */
    .info-card {
        background: #151922;
        border: 1px solid #292f3b;
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        min-height: 145px;
    }

    .info-card-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .column-pill {
        display: inline-block;
        background: #202633;
        border: 1px solid #343c4a;
        color: #dbe1e8;
        padding: 0.38rem 0.65rem;
        margin: 0.18rem 0.18rem 0.18rem 0;
        border-radius: 8px;
        font-size: 0.82rem;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #292f3b;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border-color: #252b35;
        margin: 2rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #697381;
        font-size: 0.82rem;
        padding-top: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🤖 AI JANVI</div>
    <div class="hero-subtitle">
        Upload any CSV and explore your dataset with a clean, intelligent dashboard.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# CSV UPLOAD
# =========================================================

st.markdown('<div class="section-title">📂 Upload Dataset</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Choose a CSV file to generate an instant dataset overview.</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
    label_visibility="collapsed"
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        # CSV validation
        if df.empty:
            st.error("❌ CSV file is empty. Please upload a valid CSV file.")
            st.stop()

        if len(df.columns) == 0:
            st.error("❌ No columns found in the CSV file.")
            st.stop()

        st.markdown(
            '<div class="upload-success">✅ CSV uploaded successfully</div>',
            unsafe_allow_html=True
        )

        # =====================================================
        # DATASET OVERVIEW
        # =====================================================

        st.markdown(
            '<div class="section-title">📊 Dataset Overview</div>',
            unsafe_allow_html=True
        )

        total_rows = df.shape[0]
        total_columns = df.shape[1]
        missing_values = int(df.isnull().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        col1, col2, col3, col4 = st.columns(4)

        metrics = [
            ("📌 Rows", total_rows),
            ("📋 Columns", total_columns),
            ("❌ Missing Values", missing_values),
            ("🔁 Duplicate Rows", duplicate_rows),
        ]

        for col, (label, value) in zip(
            [col1, col2, col3, col4], metrics
        ):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # =====================================================
        # DATASET PREVIEW
        # =====================================================

        st.markdown(
            '<div class="section-title">👀 Dataset Preview</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-caption">A quick look at the first rows of your uploaded data.</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=False
        )

        # =====================================================
        # COLUMN INFORMATION
        # =====================================================

        st.markdown(
            '<div class="section-title">🔍 Column Information</div>',
            unsafe_allow_html=True
        )

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            exclude="number"
        ).columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            numeric_html = "".join(
                f'<span class="column-pill">{col}</span>'
                for col in numeric_columns
            )

            if not numeric_html:
                numeric_html = '<span style="color:#8f98a5;">No numeric columns found.</span>'

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-card-title">🔢 Numeric Columns</div>
                    {numeric_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            categorical_html = "".join(
                f'<span class="column-pill">{col}</span>'
                for col in categorical_columns
            )

            if not categorical_html:
                categorical_html = '<span style="color:#8f98a5;">No text/categorical columns found.</span>'

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-card-title">📝 Categorical / Text Columns</div>
                    {categorical_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        # =====================================================
        # MISSING VALUE REPORT
        # =====================================================

        st.markdown(
            '<div class="section-title">❌ Missing Value Report</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-caption">Columns containing empty or missing values.</div>',
            unsafe_allow_html=True
        )

        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]

        if len(missing_data) == 0:
            st.success("No missing values found! ✅")
        else:
            missing_report = pd.DataFrame({
                "Column": missing_data.index,
                "Missing Values": missing_data.values,
                "Missing %": (
                    (missing_data.values / len(df)) * 100
                ).round(2)
            })

            st.dataframe(
                missing_report,
                use_container_width=True,
                hide_index=True
            )

        # =====================================================
        # STEP 10/11: ADVANCED ANALYSIS
        # =====================================================

        st.markdown(
            '<div class="section-title">🧠 Advanced AI Analysis</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-caption">'
            'The dashboard uses the actual analysis engine from app.py.'
            '</div>',
            unsafe_allow_html=True
        )

        if not BACKEND_READY:
            st.error("Could not load the analysis backend from app.py.")
            st.code(BACKEND_ERROR)
        else:
            analysis_tab, correlation_tab, outlier_tab, quality_tab, ai_tab = st.tabs(
                [
                    "💡 Insights",
                    "🔗 Correlation",
                    "📦 Outliers",
                    "🧹 Quality",
                    "🤖 Gemini AI"
                ]
            )

            # app.py automatic_insights() returns a dictionary.
            with analysis_tab:
                insights = automatic_insights(df)

                numeric_insights = insights.get("numeric_insights", {})
                categorical_insights = insights.get("categorical_insights", {})

                if numeric_insights:
                    st.markdown("**Numeric Insights**")
                    numeric_rows = []
                    for column, values in numeric_insights.items():
                        numeric_rows.append({
                            "Column": column,
                            "Count": values.get("count"),
                            "Average": values.get("average"),
                            "Median": values.get("median"),
                            "Minimum": values.get("minimum"),
                            "Maximum": values.get("maximum"),
                        })
                    st.dataframe(
                        pd.DataFrame(numeric_rows),
                        use_container_width=True,
                        hide_index=True
                    )

                if categorical_insights:
                    st.markdown("**Categorical Insights**")
                    categorical_rows = []
                    for column, values in categorical_insights.items():
                        categorical_rows.append({
                            "Column": column,
                            "Unique Values": values.get("unique_values"),
                            "Top Values": str(values.get("top_values", {}))
                        })
                    st.dataframe(
                        pd.DataFrame(categorical_rows),
                        use_container_width=True,
                        hide_index=True
                    )

                if not numeric_insights and not categorical_insights:
                    st.info("No analyzable columns found.")

            # app.py correlation_analysis() returns a dictionary containing
            # correlation_matrix and strong_relationships.
            with correlation_tab:
                correlation_result = correlation_analysis(df)
                corr_data = correlation_result.get("correlation_matrix", {})
                strong_data = correlation_result.get("strong_relationships", [])

                if not corr_data:
                    st.info(
                        correlation_result.get(
                            "message",
                            "At least two numeric columns are required for correlation analysis."
                        )
                    )
                else:
                    st.markdown("**Correlation Matrix**")
                    corr_df = pd.DataFrame(corr_data)
                    st.dataframe(
                        corr_df.round(3),
                        use_container_width=True
                    )

                    st.markdown("**Strong Relationships (|r| ≥ 0.70)**")
                    if strong_data:
                        st.dataframe(
                            pd.DataFrame(strong_data),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No strong relationships found.")

            # app.py outlier_analysis() returns a dictionary keyed by column.
            with outlier_tab:
                outlier_result = outlier_analysis(df)

                if not outlier_result:
                    st.info("No numeric columns available.")
                else:
                    outlier_rows = []
                    for column, values in outlier_result.items():
                        outlier_rows.append({
                            "Column": column,
                            "Q1": values.get("Q1"),
                            "Q3": values.get("Q3"),
                            "IQR": values.get("IQR"),
                            "Lower Bound": values.get("lower_bound"),
                            "Upper Bound": values.get("upper_bound"),
                            "Outlier Count": values.get("outlier_count"),
                            "Outlier Values": str(values.get("outlier_values", []))
                        })

                    st.dataframe(
                        pd.DataFrame(outlier_rows),
                        use_container_width=True,
                        hide_index=True
                    )

            # app.py data_quality_analysis() returns a dictionary.
            with quality_tab:
                quality_result = data_quality_analysis(df)

                quality_rows = []
                for column, values in quality_result.get("columns", {}).items():
                    quality_rows.append({
                        "Column": column,
                        "Data Type": values.get("data_type"),
                        "Missing Values": values.get("missing_values"),
                        "Missing %": values.get("missing_percentage"),
                        "Unique Values": values.get("unique_values"),
                    })

                st.markdown(
                    f"**Rows:** {quality_result.get('total_rows', 0)}  "
                    f"**Columns:** {quality_result.get('total_columns', 0)}  "
                    f"**Duplicates:** {quality_result.get('duplicate_rows', 0)}"
                )

                st.dataframe(
                    pd.DataFrame(quality_rows),
                    use_container_width=True,
                    hide_index=True
                )

            # app.py Gemini function returns text.
            with ai_tab:
                st.markdown("### 🤖 Gemini Deep Analysis")

                if st.button(
                    "✨ Generate AI Analysis",
                    type="primary",
                    use_container_width=True
                ):
                    with st.spinner("Gemini is analyzing the dataset..."):
                        try:
                            st.session_state["ai_result"] = gemini_deep_analysis(df)
                        except Exception as e:
                            st.error(f"Gemini analysis failed: {e}")

                if "ai_result" in st.session_state:
                    st.markdown(st.session_state["ai_result"])

                st.markdown("### 📄 Complete AI Report")

                if st.button(
                    "📝 Generate Complete AI Report",
                    use_container_width=True
                ):
                    with st.spinner("Generating report..."):
                        try:
                            report_result = generate_ai_analysis_report(df)
                            report_path = report_result.get("file", "")

                            if report_path and Path(report_path).exists():
                                report_text = Path(report_path).read_text(
                                    encoding="utf-8"
                                )
                            else:
                                report_text = (
                                    "Report was generated, but the report file "
                                    "could not be read."
                                )

                            st.session_state["report_text"] = report_text
                            st.session_state["report_path"] = report_path

                            st.success(
                                f"Report generated successfully: {report_path}"
                            )
                        except Exception as e:
                            st.error(f"Report generation failed: {e}")

                if "report_text" in st.session_state:
                    st.markdown("### Report Preview")
                    st.markdown(st.session_state["report_text"])

                    st.download_button(
                        "⬇️ Download AI Report",
                        data=st.session_state["report_text"],
                        file_name="ai_analysis_report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

        # =====================================================
        # STEP 11: VISUALIZATIONS
        # =====================================================

        st.markdown(
            '<div class="section-title">📈 Visualizations</div>',
            unsafe_allow_html=True
        )

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            exclude="number"
        ).columns.tolist()

        if numeric_columns:
            selected_numeric = st.selectbox(
                "Choose a numeric column",
                numeric_columns,
                key="step11_numeric_chart"
            )

            fig, ax = plt.subplots()
            ax.hist(
                df[selected_numeric].dropna(),
                bins=15
            )
            ax.set_title(
                f"Distribution of {selected_numeric}"
            )
            ax.set_xlabel(selected_numeric)
            ax.set_ylabel("Frequency")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if len(numeric_columns) >= 2:
            st.markdown("### 🔄 Numeric Relationship")

            x_col = st.selectbox(
                "X-axis",
                numeric_columns,
                key="step11_x_axis"
            )

            y_options = [
                col for col in numeric_columns
                if col != x_col
            ]

            y_col = st.selectbox(
                "Y-axis",
                y_options,
                key="step11_y_axis"
            )

            chart_df = df[[x_col, y_col]].dropna()

            fig, ax = plt.subplots()
            ax.scatter(
                chart_df[x_col],
                chart_df[y_col],
                alpha=0.7
            )
            ax.set_title(f"{x_col} vs {y_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if categorical_columns:
            st.markdown("### 🏷️ Category Distribution")

            selected_category = st.selectbox(
                "Choose a categorical column",
                categorical_columns,
                key="step11_category_chart"
            )

            counts = (
                df[selected_category]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(15)
            )

            fig, ax = plt.subplots()
            counts.plot(
                kind="bar",
                ax=ax
            )
            ax.set_title(
                f"Top Categories in {selected_category}"
            )
            ax.set_xlabel(selected_category)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # =====================================================
        # FOOTER
        # =====================================================

        st.markdown(
            '<div class="footer">AI JANVI • Step 11 Professional UI</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Unable to read this CSV file: {e}")
