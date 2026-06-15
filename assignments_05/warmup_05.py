# --- Setup ---
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

# --- API Question 1 ---
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "What is one thing that makes Python a good language for beginners?"
    }]
)

print("\n--- API Q1 ---")
print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Total Tokens Used:", response.usage.total_tokens)


# --- API Question 2 ---
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

print("\n--- API Q2 ---")
for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"\nTemperature {temp}:")
    print(response.choices[0].message.content)

# Comment:
# Lower temperature (0) gives consistent and deterministic outputs.
# Higher temperature (1.5) gives more creative but less stable results.
# Use low temperature when reproducibility matters.


# --- API Question 3 ---
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."
    }],
    n=3,
    temperature=1.0
)

print("\n--- API Q3 ---")
for i, choice in enumerate(response.choices, start=1):
    print(f"\nResponse {i}: {choice.message.content}")


# --- API Question 4 ---
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Explain how neural networks work."
    }],
    max_tokens=15
)

print("\n--- API Q4 ---")
print(response.choices[0].message.content)

# Comment:
# max_tokens limits the output length. It is useful for controlling cost,
# preventing overly long responses, and enforcing concise answers.


# --- System Question 1 ---
messages_1 = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with encouragement."
    },
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_1
)

print("\n--- System Q1 (Personality 1) ---")
print(response1.choices[0].message.content)


messages_2 = [
    {
        "role": "system",
        "content": "You are a strict senior software engineer. Be direct and concise."
    },
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_2
)

print("\n--- System Q1 (Personality 2) ---")
print(response2.choices[0].message.content)

# Comment:
# Changing the system message changes tone, depth, and personality of responses.


# --- System Question 2 ---
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

print("\n--- System Q2 ---")
print(response.choices[0].message.content)

# Comment:
# The model "remembers" Jordan because we manually included conversation history.
# The API itself is stateless.


# --- Prompt Question 1 (Zero-shot) ---
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

print("\n--- Zero-shot ---")
for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Classify sentiment as positive, negative, or mixed.\nReview: {review}"
        }]
    )
    print(f"Review {i}: {response.choices[0].message.content}")


# --- Prompt Question 2 (One-shot) ---
example = 'Review: "Fast shipping but the item arrived damaged." Sentiment: mixed'

print("\n--- One-shot ---")
for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"{example}\nReview: {review}\nSentiment:"
        }]
    )
    print(f"Review {i}: {response.choices[0].message.content}")

# Comment:
# One example improves formatting consistency compared to zero-shot.


# --- Prompt Question 3 (Few-shot) ---
prompt = """
Examples:
Review: I love this app. Sentiment: positive
Review: This is terrible. Sentiment: negative
Review: Fast shipping but damaged item. Sentiment: mixed
"""

print("\n--- Few-shot ---")
for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": prompt + f"\nReview: {review}\nSentiment:"
        }]
    )
    print(f"Review {i}: {response.choices[0].message.content}")

# Comment:
# Few-shot gives most consistent structure; zero-shot is flexible but noisy.


# --- Prompt Question 4 (Chain of Thought) ---
problem = """
A data engineer earns $85,000 per year. She gets a 12% raise,
then takes a new job that pays $7,500 more than her raised salary.
What is her final salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": problem}]
)

print("\n--- Chain of Thought ---")
print(response.choices[0].message.content)

# Comment:
# Step-by-step reasoning improves accuracy on multi-step calculations.


# --- Prompt Question 5 (Structured Output) ---
review = "I've been using this tool for three months. It handles large datasets well, but UI is clunky."

prompt = f"""
Return ONLY JSON:
{{
  "sentiment": "...",
  "confidence": 0.0,
  "reason": "..."
}}

Review: "{review}"
"""

print("\n--- Structured Output ---")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

raw = response.choices[0].message.content
print("RAW:", raw)

try:
    data = json.loads(raw)
    print("\nParsed:")
    print("Sentiment:", data["sentiment"])
    print("Confidence:", data["confidence"])
    print("Reason:", data["reason"])
except Exception:
    print("JSON parsing failed")


# --- Prompt Question 6 (Delimiters) ---
user_text = "First boil water. Add pasta. Cook for 10 minutes."

prompt = f"""
If steps exist, convert to numbered list.
Else say: No steps provided.

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\n--- Delimiters ---")
print(response.choices[0].message.content)

# Comment:
# Delimiters prevent mixing instructions with user input.


# --- Ollama Question 1 ---
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Explain what a large language model is in two sentences."
    }]
)

print("\n--- OpenAI vs Ollama ---")
print(response.choices[0].message.content)

# Ollama output (example)
"""
Ollama Output:
A large language model is an AI trained on large text datasets...
"""

# Comment:
# Local models are faster/privacy-friendly but less powerful than cloud models.
