import os
import glob
import pandas as pd
from smolagents import tool
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel
from dotenv import load_dotenv
 

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

# Tool 1: load_happiness_data

# Global dataframe
df = None
DATA_PATH = "assignments_01/outputs/merged_happiness.csv"

@tool

def load_happiness_data() -> dict:
    """
    Load the World Happiness dataset into memory.
    Tries:
    1. Load pre-merged dataset from DATA_PATH
    2. If not found, load and merge yearly CSVs from:
       assignments/resources/happiness_project/
    """
    global df

    # CASE 1: Load pre-merged file
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return {
            "shape": df.shape,
            "columns": list(df.columns)
        }
    
    # CASE 2: Fallback - load yearly CSVs and merge
    folder_path = "assignments/resources/happiness_project/"
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        return {
            "error": "No dataset found in merged path or yearly folder."
        }
    yearly_dfs = []
    for file in all_files:
        temp_df = pd.read_csv(file)
        yearly_dfs.append(temp_df)
    df = pd.concat(yearly_dfs, ignore_index=True)
    return {
        "shape": df.shape,
        "columns": list(df.columns)
    }

# Tool 2: summarize_column
@tool
def summarize_column(column: str) -> dict:
    """
    Return descriptive statistics for a single column in the loaded dataset.
    Args:
        column: Name of the column to summarize.
    """
    global df
    # Validate dataset
    if df is None:
        return {"error": "No dataset loaded."}
    # Validate column
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    try:
        return df[column].describe().to_dict()
    except Exception as e:
        return {"error": str(e)}
    
# Tool 3: compute_correlation
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation coefficient and p-value
    between two numeric columns in the loaded dataset.
    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dictionary containing:
        - col1: first column name
        - col2: second column name
        - pearson_r: Pearson correlation coefficient
        - p_value: statistical p-value
        Returns {"error": "..."} if:
        - no dataset is loaded
        - either column does not exist
        - correlation computation fails
    """
    global df
    # Validate dataset
    if df is None:
        return {"error": "No dataset loaded."}
    # Validate columns
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns not found."}
    try:
        series_1 = df[col1]
        series_2 = df[col2]
        r, p = pearsonr(series_1, series_2)
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4)
        }
    except Exception as e:
        return {"error": str(e)}
    


# Tool 4: get_top_n_countries
@tool

def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """
    Return the top N countries ranked by a given column for a specific year.
    This tool filters the global World Happiness dataset by year,
    sorts the values of the selected column in descending order,
    and returns the top N countries.

    Args:
        column: Name of the numeric column to rank (e.g., "Life Ladder").
        year: Year to filter the dataset by.
        n: Number of top countries to return (default is 5).

    Returns:
        A dictionary containing a list of the top N countries with their values:
        {
            "year": int,
            "column": str,
            "results": [
                {"country": str, "value": float},
                ...
            ]
        }
        Returns {"error": "..."} if:
        - dataset is not loaded
        - year or column is invalid
        - required data is missing
    """
    global df
    # Validate dataset
    if df is None:
        return {"error": "No dataset loaded."}
    
    # Validate column
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    
    # Try to filter year column (assumes dataset has 'year' + 'Country name')
    if "year" not in df.columns:
        return {"error": "No 'year' column found in dataset."}
    if "Country name" not in df.columns:
        return {"error": "No 'Country name' column found in dataset."}
    try:
        filtered = df[df["year"] == year]
        if filtered.empty:
            return {"error": f"No data found for year {year}."}
        sorted_df = filtered.sort_values(by=column, ascending=False).head(n)
        results = [
            {
                "country": row["Country name"],
                "value": float(row[column])
            }
            for _, row in sorted_df.iterrows()
        ]
        return {
            "year": year,
            "column": column,
            "results": results
        }
    except Exception as e:
        return {"error": str(e)}
    

# Task 2: Build the Agent
# Create the model

model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini"
)

# System prompt
SYSTEM_PROMPT = """
    You are a data analyst assistant for the World Happiness dataset.
    Use the available tools for:
    - loading data
    - summarizing columns
    - computing correlations
    - ranking countries
    Write Python code only when necessary, and prefer using tools first.
    Be concise and explain results clearly.
    """
# Create the CodeAgent
agent = CodeAgent(
    tools=[
        load_happiness_data,
        summarize_column,
        compute_correlation,
        get_top_n_countries
    ],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=[
        "pandas",
        "matplotlib.pyplot",
        "scipy.stats"
    ],
    max_steps=8,
)


# Task 3: Run Guided Queries
if __name__ == "__main__":

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Then summarize the 'Happiness score' column.",
        "What is the correlation between 'GDP per capita' and 'Happiness score'?",
        "Show me the top 5 happiest countries in 2020 ranked by 'Happiness score'.",
        "Plot 'Happiness score' over years by 'Regional indicator'.",
    ]

    # Run queries sequentially while preserving conversation memory
    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(
            query,
            reset=False
        )
        print(response)

   # =========================
    # Task 4: Custom Queries
    # =========================
    custom_queries = [
        "What is the average Happiness score by region?",
        "Compare GDP per capita and Life Ladder correlation for 2019."
    ]

    for query in custom_queries:
        print(f"\n--- Custom Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

        # Comment: Agent behavior (tool call / code generation / hybrid)
        # - First query likely triggers: tool usage (summarization/groupby) + possible code
        # - Second query likely triggers: tool usage (compute_correlation) or hybrid

# Verification Note

# Query 5 should trigger the CodeAgent to generate
# custom matplotlib code because no tool exists
# for creating multi-line regional trend plots.
#
# After running, verify that this file exists:
#
# assignments_07/outputs/happiness_by_region.png
#
# If the file was successfully created, then the
# agent correctly switched from tool usage to
# direct Python code generation.



# =========================
# Task 5: Reflection
# =========================

# 1. Statistical interpretation of Query 3 (GDP per capita vs Happiness score):
#    The agent computes Pearson correlation, which measures linear relationship strength.
#    A high positive value would indicate that countries with higher GDP per capita
#    tend to have higher happiness scores. The p-value indicates whether this
#    relationship is statistically significant (typically p < 0.05).

# 2. Any surprising response from the agent:
#    In some runs, the agent may switch between tool usage and raw Python code.
#    A surprising behavior is when the agent re-implements correlation logic in code
#    instead of using the compute_correlation tool, even though the tool is available.
#    Another possible issue is column mismatch (e.g., "Happiness score" vs "Life Ladder").

# 3. Additional tool that would improve analysis:
#    A useful additional tool would be a groupby_aggregate tool that allows:
#    - grouping by region or year
#    - computing mean/median/std automatically
#    This would reduce the need for the agent to write custom pandas code for
#    every grouped analysis (especially for regional comparisons and trends).