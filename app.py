from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# AI DATA ANALYST AGENT
# STEP 9 - DYNAMIC DATASET AGENT
# ============================================================

# ------------------------------------------------------------
# 1. Load API key safely from .env
# ------------------------------------------------------------
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Add it to your .env file."
    )

client = genai.Client(api_key=api_key)


# ------------------------------------------------------------
# 9.1 - Dynamic CSV Reading
# ------------------------------------------------------------
def load_dataset(file_path):
    """Load any CSV file into a pandas DataFrame."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Please provide a CSV file.")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The CSV file is empty.")

    return df


# Default dataset path used only when app.py is run directly.
# Streamlit will pass its uploaded DataFrame to the analysis functions.
DATASET_PATH = "data/students.csv"

# ------------------------------------------------------------
# 9.2 - Automatic Column Detection
# ------------------------------------------------------------
def detect_columns(df):
    """Automatically detect numeric and text/categorical columns."""

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    return numeric_columns, categorical_columns


# ------------------------------------------------------------
# 9.3 - Dynamic Data Analysis
# ------------------------------------------------------------
def validate_column(df, column):
    """Check whether a requested column exists."""

    if column not in df.columns:
        return False, f"Column '{column}' not found in dataset."

    return True, ""


def validate_numeric_column(df, column):
    """Check whether a requested column is numeric."""

    valid, message = validate_column(df, column)

    if not valid:
        return False, message

    if not pd.api.types.is_numeric_dtype(df[column]):
        return False, (
            f"Column '{column}' is not numeric. "
            f"Choose one of: {detect_columns(df)[0]}"
        )

    return True, ""


def get_column_info(df, column):
    """Return useful information about any dataset column."""

    valid, message = validate_column(df, column)

    if not valid:
        return message

    series = df[column]

    result = {
        "column": column,
        "data_type": str(series.dtype),
        "non_null_values": int(series.notna().sum()),
        "missing_values": int(series.isna().sum()),
        "unique_values": int(series.nunique(dropna=True)),
    }

    if pd.api.types.is_numeric_dtype(series):
        result.update({
            "minimum": float(series.min()),
            "maximum": float(series.max()),
            "average": float(series.mean()),
            "median": float(series.median()),
        })

    return result


def average_value(df, column):
    """Calculate the average of any numeric column."""

    valid, message = validate_numeric_column(df, column)

    if not valid:
        return message

    return round(float(df[column].mean()), 2)


def top_records(df, column, n=3):
    """Return top N rows based on any numeric column."""

    valid, message = validate_numeric_column(df, column)

    if not valid:
        return message

    n = max(1, min(int(n), len(df)))

    result = df.nlargest(n, column)

    return result.to_dict(orient="records")


def highest_lowest(df, column):
    """Return highest and lowest values from any numeric column."""

    valid, message = validate_numeric_column(df, column)

    if not valid:
        return message

    return {
        "column": column,
        "highest": float(df[column].max()),
        "lowest": float(df[column].min()),
    }


def data_summary(df):
    """Return statistical summary for all numeric columns."""

    numeric_columns, _ = detect_columns(df)

    if not numeric_columns:
        return "No numeric columns available for statistical summary."

    return df[numeric_columns].describe().round(2).to_dict()


def dataset_overview(df):
    """Return dynamic overview of the complete dataset."""

    numeric_columns, categorical_columns = detect_columns(df)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# ------------------------------------------------------------
# 9.4 - Dynamic Visualization
# ------------------------------------------------------------
def create_dynamic_visualizations(df, output_folder="charts"):
    """
    Automatically create useful charts based on the dataset.
    No Math/Science/Name columns are hard-coded.
    """

    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    generated_files = []

    nums, cats = detect_columns(df)

    # Chart 1: Average of all numeric columns
    if nums:
        averages = [df[col].mean() for col in nums]

        plt.figure(figsize=(10, 5))
        plt.bar(nums, averages)
        plt.title("Average Values of Numeric Columns")
        plt.xlabel("Column")
        plt.ylabel("Average")
        plt.xticks(rotation=45)
        plt.tight_layout()

        file = output_path / "dynamic_numeric_averages.png"
        plt.savefig(file)
        plt.close()

        generated_files.append(str(file))

    # Chart 2: Distribution of first numeric column
    if nums:
        column = nums[0]

        plt.figure(figsize=(8, 5))
        plt.hist(df[column].dropna(), bins=10)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()

        file = output_path / "dynamic_distribution.png"
        plt.savefig(file)
        plt.close()

        generated_files.append(str(file))

    # Chart 3: Category counts for first text/categorical column
    if cats:
        column = cats[0]
        counts = df[column].value_counts().head(10)

        plt.figure(figsize=(8, 5))
        plt.bar(counts.index.astype(str), counts.values)
        plt.title(f"Top Categories in {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()

        file = output_path / "dynamic_categories.png"
        plt.savefig(file)
        plt.close()

        generated_files.append(str(file))

    # Chart 4: Relationship between first two numeric columns
    if len(nums) >= 2:
        x = nums[0]
        y = nums[1]

        plt.figure(figsize=(8, 5))
        plt.scatter(df[x], df[y])
        plt.title(f"{x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()

        file = output_path / "dynamic_relationship.png"
        plt.savefig(file)
        plt.close()

        generated_files.append(str(file))

    if not generated_files:
        return {
            "message": "No suitable numeric or categorical columns were found.",
            "files": []
        }

    return {
        "message": "Dynamic visualizations created successfully.",
        "files": generated_files
    }


# ------------------------------------------------------------
# 9.5 - AI Agent Dataset Understanding
# ------------------------------------------------------------
def get_dataset_overview():
    return dataset_overview(df)


def get_numeric_columns():
    return detect_columns(df)[0]


def get_categorical_columns():
    return detect_columns(df)[1]


def get_column_details(column: str):
    return get_column_info(df, column)


def get_average(column: str):
    return average_value(df, column)


def get_top_records(column: str, n: int = 3):
    return top_records(df, column, n)


def get_highest_lowest(column: str):
    return highest_lowest(df, column)


def get_data_summary():
    return data_summary(df)


def get_visualizations():
    return create_dynamic_visualizations(df)


tools = [
    get_dataset_overview,
    get_numeric_columns,
    get_categorical_columns,
    get_column_details,
    get_average,
    get_top_records,
    get_highest_lowest,
    get_data_summary,
    get_visualizations,
]


# ------------------------------------------------------------
# 9.6 - Dynamic Testing + Error Handling
# ------------------------------------------------------------
def agent_tool_router(df, question):
    """
    Simple local router for testing.
    The Gemini agent below is the main intelligent router.
    """

    question_lower = question.lower()

    # Overview
    if "overview" in question_lower:
        return dataset_overview(df)

    # Summary
    if "summary" in question_lower or "statistics" in question_lower:
        return data_summary(df)

    # Visualizations
    if (
        "visualization" in question_lower
        or "visualisation" in question_lower
        or "chart" in question_lower
        or "graph" in question_lower
    ):
        return create_dynamic_visualizations(df)

    # Find a column mentioned by the user
    column = None

    for col in df.columns:
        if col.lower() in question_lower:
            column = col
            break

    if column is None:
        return (
            "I could not identify a dataset column in your question. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Average
    if "average" in question_lower or "mean" in question_lower:
        return average_value(df, column)

    # Top
    if "top" in question_lower:
        return top_records(df, column, 3)

    # Highest
    if "highest" in question_lower or "maximum" in question_lower:
        return highest_lowest(df, column)["highest"]

    # Lowest
    if "lowest" in question_lower or "minimum" in question_lower:
        return highest_lowest(df, column)["lowest"]

    # Column information
    if "column" in question_lower or "information" in question_lower:
        return get_column_info(df, column)

    return (
        "I understand the dataset, but I could not map the question "
        "to a supported analysis. Try asking for average, top, "
        "highest, lowest, overview, summary, or charts."
    )



# ============================================================
# STEP 10 - ADVANCED AI ANALYSIS + AI REPORT
# ============================================================

# ------------------------------------------------------------
# 10.1 - Automatic Insights Engine
# ------------------------------------------------------------
def automatic_insights(df):
    """Automatically calculate important statistics for numeric columns."""
    nums, cats = detect_columns(df)

    insights = {
        "numeric_insights": {},
        "categorical_insights": {},
    }

    for column in nums:
        series = pd.to_numeric(df[column], errors="coerce").dropna()

        if series.empty:
            continue

        insights["numeric_insights"][column] = {
            "count": int(series.count()),
            "average": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "minimum": round(float(series.min()), 2),
            "maximum": round(float(series.max()), 2),
        }

    for column in cats:
        series = df[column].dropna()

        insights["categorical_insights"][column] = {
            "unique_values": int(series.nunique()),
            "top_values": series.astype(str).value_counts().head(5).to_dict(),
        }

    return insights


# ------------------------------------------------------------
# 10.2 - Correlation & Relationship Analysis
# ------------------------------------------------------------
def correlation_analysis(df, threshold=0.70):
    """Find strong relationships between numeric columns."""
    nums, _ = detect_columns(df)

    if len(nums) < 2:
        return {
            "message": "At least two numeric columns are required.",
            "correlation_matrix": {},
            "strong_relationships": [],
        }

    matrix = df[nums].corr().round(2)
    strong_relationships = []

    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            value = matrix.loc[nums[i], nums[j]]

            if pd.notna(value) and abs(float(value)) >= threshold:
                strength = "positive" if value > 0 else "negative"

                strong_relationships.append({
                    "column_1": nums[i],
                    "column_2": nums[j],
                    "correlation": round(float(value), 2),
                    "relationship": f"Strong {strength} relationship",
                })

    return {
        "correlation_matrix": matrix.to_dict(),
        "strong_relationships": strong_relationships,
        "threshold": threshold,
    }


# ------------------------------------------------------------
# 10.3 - Outlier Detection using IQR
# ------------------------------------------------------------
def outlier_analysis(df):
    """Detect outliers dynamically in every numeric column using IQR."""
    nums, _ = detect_columns(df)
    results = {}

    for column in nums:
        series = pd.to_numeric(df[column], errors="coerce").dropna()

        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outliers = series[
            (series < lower_bound) | (series > upper_bound)
        ]

        results[column] = {
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outlier_count": int(outliers.count()),
            "outlier_values": [
                round(float(value), 2) for value in outliers.tolist()
            ],
        }

    return results


# ------------------------------------------------------------
# 10.4 - Data Quality Analysis
# ------------------------------------------------------------
def data_quality_analysis(df):
    """Check missing values, duplicates, data types and uniqueness."""
    quality = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {},
    }

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_percentage = (
            round((missing_count / len(df)) * 100, 2)
            if len(df) > 0 else 0.0
        )

        quality["columns"][column] = {
            "data_type": str(df[column].dtype),
            "missing_values": missing_count,
            "missing_percentage": missing_percentage,
            "unique_values": int(df[column].nunique(dropna=True)),
        }

    return quality


# ------------------------------------------------------------
# 10.5 - Gemini AI Deep Analysis
# ------------------------------------------------------------
def gemini_deep_analysis(df):
    """Send factual Python-generated analysis to Gemini for interpretation."""
    facts = {
        "dataset_overview": dataset_overview(df),
        "automatic_insights": automatic_insights(df),
        "correlation_analysis": correlation_analysis(df),
        "outlier_analysis": outlier_analysis(df),
        "data_quality": data_quality_analysis(df),
    }

    prompt = f"""
You are an expert AI Data Analyst.

Analyze the following Python-generated dataset facts.

IMPORTANT RULES:
1. Use ONLY the facts provided below.
2. Do not invent numerical values.
3. Do not invent columns, rows, correlations, or outliers.
4. If something is not available, say that it is not available.
5. Explain findings in simple professional language.
6. Do not claim correlation means causation.

Create these sections:
1. Executive Summary
2. Key Insights
3. Correlation Findings
4. Outlier Findings
5. Data Quality Findings
6. Recommendations

PYTHON-GENERATED FACTS:
{facts}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text


# ------------------------------------------------------------
# 10.6 - Complete AI Analysis Report
# ------------------------------------------------------------
def generate_ai_analysis_report(df, output_file="reports/ai_analysis_report.md"):
    """Generate and save the complete AI analysis report."""
    overview = dataset_overview(df)
    insights = automatic_insights(df)
    correlations = correlation_analysis(df)
    outliers = outlier_analysis(df)
    quality = data_quality_analysis(df)
    ai_analysis = gemini_deep_analysis(df)

    report_lines = [
        "# AI Data Analyst Report",
        "",
        "## Dataset Overview",
        f"- Rows: {overview['rows']}",
        f"- Columns: {overview['columns']}",
        f"- Numeric columns: {overview['numeric_columns']}",
        f"- Categorical columns: {overview['categorical_columns']}",
        f"- Duplicate rows: {overview['duplicate_rows']}",
        "",
        "## Automatic Insights",
        "```text",
        str(insights),
        "```",
        "",
        "## Correlation Analysis",
        "```text",
        str(correlations),
        "```",
        "",
        "## Outlier Analysis",
        "```text",
        str(outliers),
        "```",
        "",
        "## Data Quality",
        "```text",
        str(quality),
        "```",
        "",
        "## Gemini Deep Analysis",
        ai_analysis,
        "",
        "## Recommendations",
        "See the recommendations provided in the Gemini Deep Analysis above.",
        "",
    ]

    report = "\n".join(report_lines)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    return {
        "message": "AI analysis report created successfully.",
        "file": str(output_path),
    }


# Step 10 tool wrappers
def get_automatic_insights():
    return automatic_insights(df)


def get_correlation_analysis():
    return correlation_analysis(df)


def get_outlier_analysis():
    return outlier_analysis(df)


def get_data_quality_analysis():
    return data_quality_analysis(df)


def get_ai_deep_analysis():
    return gemini_deep_analysis(df)


def get_ai_analysis_report():
    return generate_ai_analysis_report(df)


# Add Step 10 tools without removing Step 9 tools.
tools.extend([
    get_automatic_insights,
    get_correlation_analysis,
    get_outlier_analysis,
    get_data_quality_analysis,
    get_ai_deep_analysis,
    get_ai_analysis_report,
])


# ------------------------------------------------------------
# Direct-run testing
# ------------------------------------------------------------
# IMPORTANT:
# This block runs ONLY when you execute:
#     python app.py
#
# When Streamlit imports this file, none of this code runs.
# Therefore Streamlit will not get stuck waiting for input().
# ------------------------------------------------------------

chat = None
df = None


def build_chat():
    """Create the Gemini agent for direct Python/CLI usage."""
    return client.chats.create(
        model="gemini-3.6-flash",
        config=types.GenerateContentConfig(
            tools=tools,
            system_instruction="""
You are an intelligent AI Data Analyst Agent.

Your job is to analyze ANY CSV dataset using the available tools.

Rules:
1. Always understand the available dataset columns before answering
   dataset-related questions.
2. Use tools whenever the answer requires actual dataset values.
3. Never guess dataset values.
4. Never assume columns such as Math, Science, Name, Salary, Age, etc.
5. Use the exact column names available in the current dataset.
6. For average, highest, lowest, and top-record questions, use a
   numeric column.
7. If a requested column does not exist, clearly tell the user and
   mention the available columns.
8. If the requested column is text/categorical and the operation
   requires numbers, explain that clearly.
9. For general dataset questions, use the overview/summary tools.
10. For chart/visualization requests, use the visualization tool.
11. After using a tool, explain the result in simple, professional
    language.
12. Do not invent rows, values, columns, or conclusions.
"""
        )
    )


def main():
    global df, chat

    df = load_dataset(DATASET_PATH)

    numeric_columns, categorical_columns = detect_columns(df)

    print("\n📊 Dataset loaded successfully!")
    print("-" * 55)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("All Columns:", df.columns.tolist())

    print("\n🔢 Numeric Columns:")
    print(numeric_columns)

    print("\n📝 Categorical/Text Columns:")
    print(categorical_columns)

    dataset_profile = dataset_overview(df)
    print("\n========== DYNAMIC DATASET PROFILE ==========")
    print(dataset_profile)

    chat = build_chat()

    print("\n" + "=" * 60)
    print("🤖 AI DATA ANALYST AGENT - STEP 10")
    print("=" * 60)

    print("\nAvailable columns:")
    print(df.columns.tolist())

    print("\nRunning automatic analysis...")

    try:
        print("\n🔍 10.1 Automatic Insights:")
        print(automatic_insights(df))

        print("\n🔗 10.2 Correlation Analysis:")
        print(correlation_analysis(df))

        print("\n🚨 10.3 Outlier Analysis:")
        print(outlier_analysis(df))

        print("\n🧹 10.4 Data Quality:")
        print(data_quality_analysis(df))

        print("\n🤖 10.5 Gemini Deep Analysis:")
        deep_analysis = gemini_deep_analysis(df)
        print(deep_analysis)

        print("\n📄 10.6 Generating AI Report...")
        report_result = generate_ai_analysis_report(df)
        print(report_result)

    except Exception as e:
        print("\n❌ Step 10 Analysis Error:")
        print(str(e))

    print("\nTry asking the AI Agent:")
    print("1. Give me automatic insights from this dataset.")
    print("2. Find strong correlations in this dataset.")
    print("3. Detect outliers in this dataset.")
    print("4. Check the data quality.")
    print("5. Give me a deep analysis of this dataset.")
    print("6. Generate the complete AI analysis report.")

    question = input("\n🤖 Ask your Data Analyst Agent: ")

    try:
        response = chat.send_message(question)

        print("\n🤖 AI Data Analyst:")
        print(response.text)

    except Exception as e:
        print("\n❌ Agent Error:")
        print(str(e))


if __name__ == "__main__":
    main()

