from prefect import flow, task
import requests
import json
from datetime import date
import os
from openai import OpenAI
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv


# =========================
# OPENAI CLIENT
# =========================
load_dotenv()
client = OpenAI()
api_key = os.getenv("OPENAI_API_KEY")

# ======================
# Constants
# ======================
ACCOUNT_URL = "https://sayedctd2026.blob.core.windows.net"
CONTAINER = "pipeline-data"

# ----------------------------
# Azure connection setup
# ----------------------------

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
credential = DefaultAzureCredential()
blob_service_client = BlobServiceClient(
    account_url=ACCOUNT_URL,
    credential=credential
)

container_client = blob_service_client.get_container_client(CONTAINER)


SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)


# =========================
# EXTRACT TASK
# =========================
@task(retries=2, retry_delay_seconds=10)
def extract_task():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=47.6062&longitude=-122.3321"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=7"
    )

    response = requests.get(url)
    response.raise_for_status()

    print("Extract: Weather data fetched successfully")

    return response.json()


# =========================
# TRANSFORM TASK
# =========================
@task
def transform_task(raw_data: dict):

    hourly = raw_data["hourly"]

    records = [
        {
            "time": t,
            "temperature_2m": temp,
            "precipitation": prec
        }
        for t, temp, prec in zip(
            hourly["time"],
            hourly["temperature_2m"],
            hourly["precipitation"]
        )
    ]

    enriched = []

    for i, r in enumerate(records):
        label = "unknown"

        if i < 24:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"{r['temperature_2m']}C, {r['precipitation']}mm"
                        }
                    ],
                    temperature=0
                )

                result = response.choices[0].message.content.strip().lower()

                if result in ["good", "marginal", "bad"]:
                    label = result

            except Exception:
                label = "unknown"

        r["condition"] = label
        enriched.append(r)

        if i % 6 == 0:
            print(f"Transform: processed {i} records")

    return enriched


# =========================
# LOAD TASK
# =========================
@task
def load_task(records: list):

    today = date.today().isoformat()
    blob_path = f"final/{today}/weather_etl.json"
    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(
        json.dumps(records, indent=2),
        overwrite=True
    )

    print(f"Uploaded {blob_path}")
    print("Blob exists:", blob_client.exists())
    return blob_path


# =========================
# FLOW
# =========================
@flow(log_prints=True)
def etl_flow():

    raw = extract_task()
    transformed = transform_task(raw)
    blob_path = load_task(transformed)

    print(f"ETL COMPLETE. File saved at: {blob_path}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    etl_flow()