# ============================================================
# --- LLMs as Transform ---
# ============================================================

# Q1

# 1. Parse "Jan 5th, 2024" → ISO date format
# Use deterministic code because date parsing follows fixed rules and does not require reasoning variability.

# 2. Classify support ticket ("card charged twice")
# Use an LLM because interpretation of meaning and intent requires semantic understanding.

# 3. Calculate average of numbers
# Use deterministic code because it is a fixed mathematical operation.

# 4. Extract company name from "Sr. Data Eng @ Acme Corp (contract)"
# Use an LLM because text structure is inconsistent and requires flexible parsing.

# 5. Check if review is >100 words
# Use deterministic code because word counting is exact and rule-based.


# ============================================================
# Q2 — Prompt Problem
# ============================================================

"""
Problem with current prompt:
The instruction "Summarize this product review in a few sentences"
produces unstructured free-text output, which is hard to store, query, or parse
in downstream systems (e.g., databases or pipelines).

This creates inconsistency because:
- Output length varies
- Format is not fixed
- Cannot reliably extract fields

Improved prompt:
"Return a JSON object with keys:
- summary (string)
- sentiment (positive, negative, or mixed)
- key_points (list of strings)

Only return valid JSON. No extra text."

This makes the output machine-readable and pipeline-friendly.
"""


# ============================================================
# Q3 — Scaling Problem
# ============================================================

"""
If each API call takes ~1 second and there are 50,000 records:

Total time = 50,000 seconds
           ≈ 13.9 hours

Practical optimization:
Use parallel processing (async requests or batching with concurrency limits).
This allows multiple API calls to run simultaneously, dramatically reducing total runtime
without changing the model.
"""


# ============================================================
# --- Azure OpenAI ---
# ============================================================

# Q1
"""
Two reasons organizations use Azure OpenAI:

1. Data compliance and governance:
   Azure allows companies to keep data within their own cloud environment,
   which is important for industries like healthcare and finance.

2. Enterprise security and access control:
   Azure integrates with Active Directory, role-based access control,
   and corporate security policies.
"""


# Q2
"""
AzureOpenAI client requires:

1. azure_endpoint:
   The specific Azure OpenAI resource URL where the service is hosted.

2. api_version:
   The version of the Azure OpenAI API being used (e.g., "2024-02-15-preview").

3. azure_ad_token OR credential method:
   Authentication method used to access Azure resources (often via Azure Active Directory).

These replace the standard OpenAI-only setup and connect to a deployed Azure resource.
"""


# Q3
"""
In AzureOpenAI, the "model" parameter does NOT take names like "gpt-4o-mini".

Instead, it takes:
- The deployment name created inside Azure OpenAI Studio

You find it in:
Azure Portal → Azure OpenAI resource → Deployments section

This deployment name maps internally to a model like GPT-4o-mini or GPT-4.
"""