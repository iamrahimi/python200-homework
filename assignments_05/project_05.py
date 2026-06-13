from dotenv import load_dotenv
from openai import OpenAI
import json

# Load API key
load_dotenv()

# Initialize client
client = OpenAI()

# Task 1: Setup and System Prompt 

load_dotenv()
client = OpenAI()

# Task 5: The Chatbot Loop
# System prompt
YOUR_SYSTEM_PROMPT = """
You are a professional job application coach.

You help users improve:
- Resumes
- Cover letters
- Job application materials
- Interview preparation

Guidelines:
- Stay focused on career and application-related topics
- Give professional, specific, and practical advice
- Do not invent fake experience or qualifications
- Encourage users to review and edit all AI-generated content before submitting it
- Acknowledge that industry expectations may vary and users should use their own judgment
"""

# Helper function
# Task 1: Setup and System Prompt
def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )

    return response.choices[0].message.content


# Resume bullet rewriter
# Task 2: Bullet Point Rewriter
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach.

    Rewrite the resume bullet points below to be:
    - More specific
    - Results-oriented
    - Professional
    - Written with strong action verbs

    Do not invent fake achievements or metrics.

    Respond ONLY with valid JSON.

    Format:
    [
      {{
        "original": "...",
        "improved": "..."
      }}
    ]

    Resume bullets:
    ```
    {bullet_text}
    ```
    """

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = get_completion(messages)

    try:
        rewritten = json.loads(response)
        return rewritten

    except json.JSONDecodeError:
        print("Failed to parse the model response.")
        return []


# Cover letter generator
# Task 3: Cover Letter Generator
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.

    The paragraph should:
    - Be 3-5 sentences
    - Sound confident and specific
    - Avoid clichés and generic phrases
    - Connect previous experience to the new role
    - Not invent qualifications

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.

    Opening:
    After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.

    Opening:
    I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions.

    Now write a new opening paragraph.

    Role: {job_title}
    Background: {background}

    Opening:
    """

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = get_completion(messages)

    return response

# Moderation check
# Task 4: Moderation Check
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print("\nJob Application Helper:")
        print("Your message was flagged by the safety system.")
        print("Please rephrase your request.\n")
        return False

    return True

# Main chatbot loop
def run_chatbot():

    # 1. Initialize conversation history
    messages = [
        {"role": "system", "content": YOUR_SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:

        user_input = input("You: ").strip()

        # 2. Exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Moderation check
        if not is_safe(user_input):
            continue

        # 5. Resume bullet rewriting
        if "bullet" in user_input.lower() or "resume" in user_input.lower() or user_input == "1":

            print("\nJob Application Helper:")
            print("Paste your bullet points below, one per line.")
            print("Type 'DONE' when finished.\n")

            raw_bullets = []

            while True:
                line = input().strip()

                if line.upper() == "DONE":
                    break

                if line:
                    raw_bullets.append(line)

            rewritten = rewrite_bullets(raw_bullets)

            print("\nImproved Resume Bullets:\n")

            for item in rewritten:
                print(f"Original : {item['original']}")
                print(f"Improved: {item['improved']}")
                print("-" * 60)

        # 6. Cover letter generator
        elif "cover letter" in user_input.lower() or user_input == "2":

            job_title = input(
                "Job Application Helper: What is the job title? "
            ).strip()

            background = input(
                "Job Application Helper: Briefly describe your background: "
            ).strip()

            result = generate_cover_letter(job_title, background)

            print("\nJob Application Helper:\n")
            print(result)

        elif user_input == "3":
            print("Ask me any career or job application question.\n")
            continue
        # 7. Regular chatbot conversation
        else:

            # Add user message
            messages.append({
                "role": "user",
                "content": user_input
            })

            # Get assistant reply
            reply = get_completion(messages)

            # Print assistant reply
            print(f"\nJob Application Helper: {reply}\n")

            # Add assistant message
            messages.append({
                "role": "assistant",
                "content": reply
            })


# Run chatbot
if __name__ == "__main__":
    run_chatbot()


# Task 6: Ethics Reflection

# ============================================================
# Reflection — Option A: Comment Block
# ============================================================

# One problem with AI job application tools is that they may give
# biased advice because the model was trained on data written mostly
# by certain industries, cultures, or communication styles. This could
# favor people who write in a more formal Western business style and
# may not work as well for people from different backgrounds or career paths.

# Another risk is that users might copy and submit AI-generated content
# directly without reviewing it carefully. The bot could generate wording
# that sounds repetitive, generic, or inaccurate for the specific job.
# In some cases, it might even accidentally exaggerate experience or use
# phrases that do not match the user's real communication style.

# One guardrail I would add is a stronger review reminder before showing
# final outputs. For example, the application could display a warning
# telling users to fact-check, personalize, and edit all generated content
# before sending it to employers. I would also keep moderation filters
# enabled to reduce harmful or inappropriate requests.