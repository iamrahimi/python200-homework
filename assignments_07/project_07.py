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
    model_id="gpt-4o-mini"
)

# System prompt
SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
IMPORTANT:
- load_happiness_data returns a dictionary, NOT a dataframe.
- happiness_data only contains:
    data["shape"]
    data["columns"]
- NEVER use:
    happiness_data.shape
    happiness_data['GDP per capita']
- The real dataframe is stored in the global variable df.
- Use df['column_name'] for dataframe operations.
Use the available tools for:
- loading data
- summarizing columns
- computing correlations
- ranking countries
Write Python code directly only when the tools are not sufficient.
Be concise and student-friendly in your responses.
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