# ============================================================
# --- Setup and System Prompt ---
# ============================================================

from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


# System Prompt
YOUR_SYSTEM_PROMPT = """
You are a professional job application coach.

You help users improve:
- Resume bullet points
- Cover letters
- Job application materials
- Interview preparation

Rules:
- Stay focused on job application topics only
- Always remind users to review and edit outputs before submitting anywhere
- Acknowledge that you may not know specific industry standards
- Encourage users to use their own judgment
"""

# Comment:
# I made the system prompt specific to job applications only so the model
# does not drift into unrelated topics and stays consistent as a career coach.


# ============================================================
# --- Task 2: Bullet Point Rewriter ---
# ============================================================

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach.

    Rewrite the bullets to be:
    - More specific
    - Results-oriented
    - Professional
    - Strong action verbs

    Do NOT invent new facts.

    Return ONLY valid JSON like this:
    [
      {{
        "original": "...",
        "improved": "..."
      }}
    ]

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    response = get_completion([{"role": "user", "content": prompt}])

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("JSON parsing failed. Raw output:")
        print(response)
        return []


# ============================================================
# --- Task 3: Cover Letter Generator ---
# ============================================================

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter openings for career changers.

    Use the style from these examples:

    Example 1:
    Role: Data Analyst at healthcare nonprofit
    Background: Nurse for 7 years, data bootcamp.
    Opening: Combines clinical experience with data analysis skills...

    Example 2:
    Role: Software Engineer at fintech startup
    Background: Banking operations + self-taught Python
    Opening: Combines domain expertise with technical transition...

    Now write:

    Role: {job_title}
    Background: {background}
    Opening:
    """

    return get_completion([{"role": "user", "content": prompt}])


# ============================================================
# --- Task 4: Moderation ---
# ============================================================

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print("\n⚠️ Message blocked: Please rephrase your request.\n")
        return False

    return True


# ============================================================
# --- Task 5: Chatbot Loop ---
# ============================================================

def run_chatbot():

    messages = [
        {"role": "system", "content": YOUR_SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("1. Rewrite resume bullets")
    print("2. Generate cover letter")
    print("3. Chat with assistant")
    print("Type 'quit' to exit\n")

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("Good luck with your applications!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        # ---------------- Resume bullets ----------------
        if "bullet" in user_input.lower() or "resume" in user_input.lower():

            print("Enter bullets (type DONE when finished):")
            raw_bullets = []

            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                raw_bullets.append(line)

            result = rewrite_bullets(raw_bullets)

            for item in result:
                print("\nOriginal:", item["original"])
                print("Improved:", item["improved"])

        # ---------------- Cover letter ----------------
        elif "cover letter" in user_input.lower():

            job_title = input("Job title: ").strip()
            background = input("Background: ").strip()

            result = generate_cover_letter(job_title, background)
            print("\nCover Letter:\n", result)

        # ---------------- Normal chat ----------------
        else:

            messages.append({"role": "user", "content": user_input})

            reply = get_completion(messages)

            print("\nAssistant:", reply)

            messages.append({"role": "assistant", "content": reply})


# Run chatbot
if __name__ == "__main__":
    run_chatbot()


# ============================================================
# --- Task 6: Ethics Reflection ---
# ============================================================

"""
One issue with AI job tools is bias in training data. The model may favor
Western corporate communication styles and disadvantage other cultural or
professional writing styles.

Another risk is users submitting AI-generated text without reviewing it.
This can lead to inaccurate or exaggerated job descriptions that don't match
real experience.

A useful guardrail would be a built-in warning before final output reminding
users to verify accuracy and personalize content before sending it to employers.
"""
