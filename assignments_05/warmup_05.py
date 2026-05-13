import os
from tokenize import Comment
from dotenv import load_dotenv
from openai import OpenAI
from streamlit import json
import json as json_module
import json

load_dotenv()
client = OpenAI()
api_key = os.getenv("OPENAI_API_KEY")


# API Question 1


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?"
        }
    ]
)

# Extract response text
message = response.choices[0].message.content

# Print results with labels
print("Model Response:\n", message)
print("\nModel Name:\n", response.model)
print("\nTotal Tokens Used:\n", response.usage.total_tokens)



# API Question 2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )

    print(f"\n--- Temperature: {temp} ---")
    print(response.choices[0].message.content)


# API Question 3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."
    }],
    n=3,
    temperature=1.0
)

# Print all 3 completions
for i, choice in enumerate(response.choices, start=1):
    print(f"\n--- Response {i} ---")
    print(choice.message.content)


# API Question 4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Explain how neural networks work."
    }],
    max_tokens=15
)

print("Response:")
print(response.choices[0].message.content)

# System Question 1


# Version 1: Encouraging tutor

messages_1 = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_1
)

print("=== Personality 1 ===")
print(response1.choices[0].message.content)



# Version 2: Different personality

messages_2 = [
    {
        "role": "system",
        "content": "You are a strict but brilliant senior software engineer. You are concise, direct, and do not sugarcoat explanations."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_2
)

print("\n=== Personality 2 ===")
print(response2.choices[0].message.content)


# System Question 2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(response.choices[0].message.content)


# Prompt Question 1 — Zero-Shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = (
    "Classify the sentiment of each review as positive, negative, or mixed. "
    "Return only the classification."
)

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"{prompt}\n\nReview: {review}"}
        ]
    )

    print(f"Review {i}: {response.choices[0].message.content}")

# Prompt Question 2 — One-Shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

example = """
    Review: "Fast shipping but the item arrived damaged."
    Sentiment: mixed
"""

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify the sentiment of each review as positive, negative, or mixed.\n\n"
                    "Example:\n"
                    f"{example}\n"
                    f"Review: \"{review}\"\n"
                    "Sentiment:"
                )
            }
        ]
    )

    print(f"Review {i}: {response.choices[0].message.content}")

#Yes — adding one example (few-shot prompting) usually changes the output in a noticeable way:
# What changed compared to zero-shot (Q1):
# More consistent formatting
# The model is more likely to output just: positive, negative, or mixed
# Less variation in wording
# Without an example, the model might respond:
# This is a positive review
# Positive sentiment
# With an example, it tends to copy the format exactly

# Prompt Question 3 — Few-Shot

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = """
    Classify the sentiment of each review as positive, negative, or mixed.

    Examples:
    Review: "I love how fast and easy the app is to use."
    Sentiment: positive

    Review: "The app is terrible and keeps freezing."
    Sentiment: negative

    Review: "Fast shipping but the item arrived damaged."
    Sentiment: mixed
"""

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt + f"\nReview: \"{review}\"\nSentiment:"
            }
        ]
    )

    print(f"Review {i}: {response.choices[0].message.content}")
# Comment (Zero-shot vs One-shot vs Few-shot)
# Zero-shot (no examples):
# Flexible but sometimes inconsistent formatting
# Good for quick tasks and general classification
# One-shot (1 example):
# Improves format consistency
# Helps model understand expected output style
# Few-shot (3+ examples):
# Most reliable structure and consistency
# Better accuracy and alignment with task expectations
# Slightly more token cost

# Prompt Question 4 — Chain of Thought

problem = """
    Solve the following problem step by step and clearly show reasoning before the final answer.

    A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
    takes a new job that pays $7,500 more per year than her post-raise salary.

    What is her final annual salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": problem}
    ]
)

print(response.choices[0].message.content)
# Comment (Why step-by-step reasoning helps)

# Asking the model to reason step by step improves accuracy because:

# It forces the model to break down multi-step calculations
# Reduces risk of skipping or combining steps incorrectly
# Makes intermediate values explicit (like salary after raise)
# Helps the model “self-check” before producing the final answer

# Prompt Question 5 — Structured Output
review = "I've been using this tool for three months. It handles large datasets well, but the UI is clunky and the export options are limited."

prompt = f"""
Analyze the review and return ONLY valid JSON with:
- sentiment (positive, negative, or mixed)
- confidence (a float from 0 to 1)
- reason (one sentence)

Review: "{review}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

raw_output = response.choices[0].message.content

print("RAW RESPONSE:")
print(raw_output)

# Parse JSON safely
try:
    data = json.loads(raw_output)

    print("\nPARSED OUTPUT:")
    print("Sentiment:", data["sentiment"])
    print("Confidence:", data["confidence"])
    print("Reason:", data["reason"])

except json.JSONDecodeError:
    print("\nFailed to parse JSON. Raw output for debugging:")
    print(raw_output)

# Why this pattern is important:
# LLMs sometimes return:
# extra text
# markdown formatting
# or slightly invalid JSON
# The try/except block helps:
# Prevent your program from crashing
# Let you debug bad model formatting
# Handle real-world API unpredictability safely


# Prompt Question 6 — Delimiters


# 1st input (has steps)

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt_1 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt_1}]
)

print("=== Instruction Text Result ===")
print(response1.choices[0].message.content)



# 2nd input (no instructions)

non_instruction_text = "The sunset over the mountains was beautiful and calming to watch from the distance."

prompt_2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{non_instruction_text}```
"""

response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt_2}]
)

print("\n=== Non-Instruction Text Result ===")
print(response2.choices[0].message.content)

# Comment (Concept)
# Why delimiters (like ) are useful:

# They help prevent a common problem called prompt injection or instruction confusion, where:

# The model might mix user content with system instructions
# It may misinterpret where instructions end and data begins
# It could accidentally follow instructions inside the user input
# What delimiters solve:
# Clearly separate instructions vs data
# Reduce ambiguity in parsing input
# Improve reliability in structured tasks


# Ollama Question 1

# OpenAI API response

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Explain what a large language model is in two sentences."
    }]
)

print("=== OpenAI Response ===")
print(response.choices[0].message.content)



# Ollama output (paste from terminal)

"""
=== Ollama (qwen3:0.6b) Output ===
A large language model is a type of artificial intelligence system trained on vast amounts of text data to understand and generate human-like language. It predicts the next word in a sentence based on patterns it has learned from training data.
"""



# Comparison comment

"""
Comparison:
- OpenAI response is usually more detailed, polished, and accurate in tone.
- Ollama (qwen3:0.6b) is shorter, simpler, and sometimes less refined or less context-aware.

Advantage of local models:
- Works offline and protects privacy (no data sent to external servers).

Disadvantage:
- Lower performance compared to large cloud models and requires local compute resources.
"""