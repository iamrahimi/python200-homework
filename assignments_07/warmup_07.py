from openai import OpenAI
import json
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from scipy.stats import pearsonr
from smolagents import tool
from typing import Dict, Union
from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
# --- Lesson 02: Tool Definitions and the ReAct Loop ---

# Q1

# Function definition
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# JSON schema dictionary for the function
celsius_to_fahrenheit_schema = {
    "name": "celsius_to_fahrenheit",
    "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
    "parameters": {
        "type": "object",
        "properties": {
            "celsius": {
                "type": "number",
                "description": "Temperature in Celsius"
            }
        },
        "required": ["celsius"]
    }
}

# Print the schema
print("Function Schema:")
print(celsius_to_fahrenheit_schema)
print("\nFunction Calls:")

# Direct function calls
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

# Q2

# Prediction:
# Calling run_agent("Convert 100 degrees Celsius to Fahrenheit")
# will NOT trigger a tool call because the only available tool
# is get_current_time, which is unrelated to temperature conversion.
#
# The model can answer the conversion directly from its own knowledge,
# so no tool is needed.
#
# Expected API calls:
# 1 API call total:
# - One call to the model to generate the final response.
# - No second API call because no tool execution happens.

# Tool function
def get_current_time():
    """Return the current local time."""
    return datetime.now().strftime("%H:%M:%S")

# Tool schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# Simple ReAct-style agent
def run_agent(user_input):
    messages = [{"role": "user", "content": user_input}]
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools
    )
    message = response.choices[0].message

    # Check if the model wants to call a tool
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        if tool_name == "get_current_time":
            result = get_current_time()
            messages.append(message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
            second_response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages
            )
            return second_response.choices[0].message.content

    # If no tool was needed
    return message.content

# Call the agent
result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print("Agent Result:")
print(result)
print("\nWas the prediction correct?")
print("Yes. No tool call was triggered because the available tool")
print("only handled time requests, not temperature conversion.")


# Q3

# Tool Functions
def get_current_time():
    """Return the current local time."""
    return datetime.now().strftime("%H:%M:%S")
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# Tool Schemas
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "celsius_to_fahrenheit",
            "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "celsius": {
                        "type": "number",
                        "description": "Temperature in Celsius"
                    }
                },
                "required": ["celsius"]
            }
        }
    }
]

# ReAct Agent
def run_agent(user_input):
    messages = [{"role": "user", "content": user_input}]
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools
    )
    message = response.choices[0].message
    # Check for tool calls
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        # Parse arguments
        arguments = json.loads(tool_call.function.arguments)
        # Dispatch tool
        if tool_name == "get_current_time":
            result = get_current_time()
        elif tool_name == "celsius_to_fahrenheit":
            result = celsius_to_fahrenheit(arguments["celsius"])
        else:
            result = "Unknown tool."
        # Add assistant tool call message
        messages.append(message)
        # Add tool result
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })
        # Second API call after tool execution
        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )
        return second_response.choices[0].message.content
    # No tool call needed
    return message.content
# Test Queries

response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# A tool WAS called here because the user explicitly requested
# a Celsius-to-Fahrenheit conversion, which matches the
# celsius_to_fahrenheit tool.
response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# A tool was likely NOT called here because the model can answer
# this general knowledge question directly without needing either tool.

# Lesson 03: Multi-Tool Agent
# Q4

# CsvManager Class
class CsvManager:
    def __init__(self):
        self.df = None
    def load_csv(self, file_path: str):
        """Load a CSV file into a pandas DataFrame."""
        try:
            self.df = pd.read_csv(file_path)
            return {
                "status": "success",
                "columns": list(self.df.columns),
                "rows": len(self.df)
            }
        except Exception as e:
            return {"error": str(e)}
    def preview_csv(self, n: int = 5):
        """Preview the first n rows of the loaded CSV."""
        if self.df is None:
            return {"error": "No CSV loaded."}
        return self.df.head(n).to_dict(orient="records")
    def summarize_column(self, column_name: str):
        """Return summary statistics for a numeric column."""
        if self.df is None:
            return {"error": "No CSV loaded."}
        if column_name not in self.df.columns:
            return {"error": f"Column '{column_name}' not found."}
        try:
            summary = self.df[column_name].describe()
            return {
                "column": column_name,
                "count": round(summary["count"], 4),
                "mean": round(summary["mean"], 4),
                "std": round(summary["std"], 4),
                "min": round(summary["min"], 4),
                "max": round(summary["max"], 4)
            }
        except Exception as e:
            return {"error": str(e)}

    # NEW METHOD FOR Q4
    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        if self.df is None:
            return {"error": "No CSV loaded."}
        if col1 not in self.df.columns:
            return {"error": f"Column '{col1}' not found."}
        if col2 not in self.df.columns:
            return {"error": f"Column '{col2}' not found."}
        try:
            r, p = pearsonr(self.df[col1], self.df[col2])
            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(r, 4),
                "p_value": round(p, 4)
            }
        except Exception as e:
            return {"error": str(e)}

# Create Manager Instance
csv_manager = CsvManager()

# Tool Schemas
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file into memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "preview_csv",
            "description": "Preview the first few rows of the CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of rows to preview"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_column",
            "description": "Get summary statistics for a numeric column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column_name": {
                        "type": "string",
                        "description": "Name of the column"
                    }
                },
                "required": ["column_name"]
            }
        }
    },

    # NEW TOOL SCHEMA FOR Q4
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation between two numeric columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "First column name"
                    },
                    "col2": {
                        "type": "string",
                        "description": "Second column name"
                    }
                },
                "required": ["col1", "col2"]
            }
        }
    }
]

# Node Tools Mapping
node_tools = {
    "load_csv": csv_manager.load_csv,
    "preview_csv": csv_manager.preview_csv,
    "summarize_column": csv_manager.summarize_column,
    # NEW TOOL ENTRY FOR Q4
    "compute_correlation": csv_manager.compute_correlation
}

# Agent Loop
def run_agent_cycle(user_input, max_rounds=5):
    messages = [{"role": "user", "content": user_input}]
    for i in range(max_rounds):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema
        )
        message = response.choices[0].message

        # If no tool call, return final response
        if not message.tool_calls:
            return message.content
        messages.append(message)

        # Execute tool calls
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool_function = node_tools.get(tool_name)
            if tool_function:
                result = tool_function(**arguments)
            else:
                result = {"error": "Unknown tool"}
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
    return "Max tool rounds reached."


# Q5

# System Prompt
SYSTEM_PROMPT = """
You are a CSV analysis assistant.
You can:
- Load CSV files
- Preview CSV data
- Summarize columns
- Compute correlations between numeric columns
Always use tools when needed.
"""

# Updated Agent Cycle
def run_agent_cycle(messages, user_input, max_rounds=5):
    # Add user message to existing conversation
    messages.append({
        "role": "user",
        "content": user_input
    })

    for i in range(max_rounds):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema
        )
        message = response.choices[0].message

        # If the model gives a final answer
        if not message.tool_calls:
            return message.content

        # Store assistant message
        messages.append(message)

        # Execute each requested tool
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool_function = node_tools.get(tool_name)
            if tool_function:
                result = tool_function(**arguments)
            else:
                result = {"error": "Unknown tool"}
            # Add tool response back into conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
    return "Max tool rounds reached."

# Recreate Lesson Scenario
messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

result = run_agent_cycle(
    messages,
    "Load assignments_07/resources/bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
)
print("Final Agent Response:")
print(result)
# With the new compute_correlation tool added,
# the agent should now successfully:
# 1. Load the CSV
# 2. Compute the Pearson correlation
# 3. Return the final analysis response
#
# The previous tool-round failure should no longer occur.


# Q6

# In the ReAct loop:
#
# - "system" messages contain the agent's instructions and behavior rules.
# - "user" messages contain the user's requests or questions.
# - "assistant" messages contain the model's reasoning and tool-call decisions.
# - "tool" messages contain the outputs returned by external tools/functions.
#
# Printing the full messages list helps visualize the full
# ReAct conversation cycle between the model and the tools.
print("\nFull Messages History:")
print(json.dumps(messages, indent=2, default=str))


# --- Lesson 04: smolagents ---

# Q7

# Smolagents Tool Wrapper
@tool
def compute_correlation(col1: str, col2: str) -> Dict[str, Union[str, float]]:
    """
    Compute the Pearson correlation between two columns
    in the currently loaded CSV DataFrame.
    Args:
        col1: The first numeric column name.
        col2: The second numeric column name.
    """
    return csv_manager.compute_correlation(col1, col2)

# Print auto-generated description
print(compute_correlation.description)

# Comparison Comment

#
# In Q4, we manually wrote a JSON schema that included:
# - tool name
# - description
# - parameter types
# - required fields
#
# With smolagents, the @tool decorator automatically generates
# much of this metadata by inspecting:
# - the function name
# - type hints
# - the docstring
#
# smolagents still depends on the developer to provide:
# - clear function names
# - accurate parameter type hints
# - detailed docstrings and argument explanations
#
# Better docstrings and type annotations produce better
# tool descriptions for the LLM automatically.



#Q8 

from typing import Dict, Any

@tool
def load_csv(file_path: str) -> Dict[str, Any]:
    """
    Load a CSV file into memory.

    Args:
        file_path (str): Path to the CSV file to load.
    """
    return csv_manager.load_csv(file_path)


@tool
def preview_csv(n: int = 5) -> Dict[str, Any]:
    """
    Preview the first rows of the CSV.

    Args:
        n (int): Number of rows to display.
    """
    return csv_manager.preview_csv(n)

@tool
def summarize_column(column_name: str) -> Dict[str, Any]:
    """
    Summarize a numeric column.

    Args:
        column_name (str): Name of the column to summarize.
    """
    return csv_manager.summarize_column(column_name)

# Model
model = OpenAIServerModel(
    model_id="gpt-4.1-mini",
    api_key=api_key
)

shared_tools = [
    load_csv,
    preview_csv,
    summarize_column,
    compute_correlation
]

# ToolCallingAgent
tool_agent = ToolCallingAgent(
    tools=shared_tools,
    model=model
)

# CodeAgent
code_agent = CodeAgent(
    tools=shared_tools,
    model=model
)

# Prompt

prompt = "Load assignments_07/resources/bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

# Run both agents

response_tool = tool_agent.run(prompt)
response_code = code_agent.run(
    prompt,
    additional_args={"csv_manager": csv_manager}
)
print("ToolCallingAgent Response:")
print(response_tool)
print("\nCodeAgent Response:")
print(response_code)

# Analysis

"""
Comparison:

1. ToolCallingAgent:
- Typically decides between available tools only.
- It does NOT directly write full Python plotting code.
- So it likely produced a tool-based or partial structured response.
- It may NOT explicitly control visualization details like "green dots".

2. CodeAgent:
- Generates and executes Python code.
- It can directly use matplotlib (or similar) to create plots.
- It is much more likely to correctly set:
  - scatter plot
  - avg_heart_rate vs duration_min
  - marker color = green

Answer to questions:
- Did ToolCallingAgent change dot color?
  → Usually NO, because it doesn't directly control plotting code.
- Did CodeAgent change dot color?
  → YES, because it writes and executes the actual plotting code.

What this reveals:
- ToolCallingAgent is better for structured tool selection tasks (API calls, retrieval, math tools).
- CodeAgent is better for tasks requiring custom computation, visualization, and flexible logic.
- CodeAgent has higher expressive power because it writes executable code, not just tool calls.
"""



# Q9
"""
Q9 Reflection:
1. When is ToolCallingAgent better than CodeAgent?
A ToolCallingAgent is better for tasks that are:
- Well-defined
- Have strict, pre-built operations (tools)
- Require safety and structure over flexibility

Example task:
"Get the current time, fetch weather data, or compute correlation using a fixed dataset tool."
Why this fits tool-based approach:
Because the action space is limited and predefined. The agent only needs to choose
the correct tool and supply arguments — not invent new logic or code.
This reduces errors and makes outputs more predictable and reliable.
2. Risk of CodeAgent that does NOT apply to ToolCallingAgent:
A CodeAgent dynamically generates and executes Python code.
One major risk:
- It can generate incorrect, unsafe, or unintended code that still executes.

For example:
- It might overwrite files
- Perform unintended computations
- Introduce logic bugs that are harder to trace
- Potentially run harmful system commands (in unsafe environments)

ToolCallingAgent avoids this because:
- It can ONLY call predefined tools
- It cannot invent arbitrary execution steps
- Its behavior is constrained by the tool interface

Summary:
- ToolCallingAgent = safe, constrained, predictable
- CodeAgent = flexible, powerful, but higher risk due to free-form code execution
"""