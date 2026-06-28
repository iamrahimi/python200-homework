import os
import glob
import pandas as pd
from smolagents import tool
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel
from dotenv import load_dotenv

# -------------------------
# Global dataset
# -------------------------
df = None
DATA_PATH = "assignments_01/outputs/merged_happiness.csv"

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")


# -------------------------
# Tool 1
# -------------------------
@tool
def load_happiness_data() -> dict:
    """
    Load the World Happiness dataset into memory.

    The function loads a merged dataset from DATA_PATH. If the file does not exist,
    it loads and merges yearly CSV files from assignments/resources/happiness_project/.

    Stores the final dataset in the global variable `df`.

    Returns:
        dict: Contains dataset shape and column names.
    """
    global df

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        import glob
        files = glob.glob("assignments_11/resources/happiness_project/*.csv")
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)

    return {"shape": df.shape, "columns": list(df.columns)}


# -------------------------
# Tool 2
# -------------------------
@tool
def summarize_column(column: str) -> dict:
    """
    Return descriptive statistics for a numeric column.

    Args:
        column (str): Column name to summarize.

    Returns:
        dict: Summary statistics or error message.
    """
    global df

    if df is None:
        return {"error": "Data not loaded"}

    if column not in df.columns:
        return {"error": "Column not found"}

    return df[column].describe().to_dict()


# -------------------------
# Tool 3
# -------------------------
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute Pearson correlation and p-value between two numeric columns.

    Args:
        col1 (str): First column.
        col2 (str): Second column.

    Returns:
        dict: correlation results or error message.
    """
    global df

    if df is None:
        return {"error": "Data not loaded"}

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "Column not found"}

    clean = df[[col1, col2]].dropna()

    r, p = pearsonr(clean[col1], clean[col2])

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(r, 4),
        "p_value": round(p, 4),
    }


# -------------------------
# Tool 4
# -------------------------
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """
    Return top N countries by a column for a specific year.

    Args:
        column (str): Column to rank by.
        year (int): Year filter.
        n (int): Number of results.

    Returns:
        dict: Top countries or error message.
    """
    global df

    if df is None:
        return {"error": "Data not loaded"}

    if column not in df.columns:
        return {"error": "Column not found"}

    if "year" not in df.columns:
        return {"error": "Year column not found"}

    subset = df[df["year"] == year].dropna(subset=[column])

    top = subset.sort_values(column, ascending=False).head(n)

    return top[["country", column]].to_dict(orient="records")


# -------------------------
# Agent setup
# -------------------------
model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini"
)

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use tools for loading, summarizing, correlations, and ranking.
Write Python code only when tools are insufficient (e.g., plotting).
Be concise and student-friendly.
"""

agent = CodeAgent(
    tools=[
        load_happiness_data,
        summarize_column,
        compute_correlation,
        get_top_n_countries
    ],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


# -------------------------
# Task 3 queries
# -------------------------
queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to assignments_07/outputs/happiness_by_region.png.",
]

if __name__ == "__main__":

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)


    # -------------------------
    # Task 4 - Custom Queries
    # -------------------------

    my_query_1 = "Which region has the highest average happiness_score?"
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)

    # Comment:
    # This query triggered tool use and possibly code generation depending on execution.

    my_query_2 = "Plot GDP per capita vs happiness score and show correlation trend."
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)

    # Comment:
    # This query triggered code generation because it required visualization.


# -------------------------
# Task 5 - Reflection
# -------------------------
"""
1. Query 3 communicated statistical significance by comparing the p-value
   to the threshold of 0.05. If p < 0.05, the result was considered statistically
   significant; otherwise, it was not.

2. A surprising capability was the agent’s ability to generate and execute Python
   code for statistical analysis and visualization without explicit instructions.
   A limitation was occasional incorrect assumptions about column names or types.

3. A useful additional tool would be a data validation tool to check missing values,
   incorrect data types, and schema consistency before analysis.
"""