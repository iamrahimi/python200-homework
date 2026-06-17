# Step 6: Reflect
"""
Reflection:

Using an LLM to classify weather conditions for outdoor running was useful, but it may not be necessary for this task. 
Since the weather data already includes temperature and precipitation values, a rule-based approach could make the same decisions more quickly and consistently. 
The advantage of using an LLM is that it can provide more flexible and human-like reasoning when evaluating conditions. However, a rule-based solution would be simpler, easier to understand, and less expensive to run.
"""


import json
import os
from datetime import date
from azure.storage.blob import BlobServiceClient
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
client = OpenAI()
api_key = os.getenv("OPENAI_API_KEY")

# video_url https://youtu.be/cUepRRf1mds

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
# Step 1: Read
def read_back(container_client, blob_path):

    """
    Download blob, save locally, parse JSON, and reshape hourly data into records.
    """
    try:
        # Try Azure blob first
        blob_client = container_client.get_blob_client(blob_path)
        if not blob_client.exists():
            raise FileNotFoundError(f"Blob {blob_path} not found")
        
        download_data = blob_client.download_blob().readall()
        weather_json = json.loads(download_data.decode("utf-8"))
        print("Loaded weather data from Azure Blob Storage.")

    except Exception as e:
        print(f"Could not load blob: {e}")
        print("Using fallback dataset.")

        with open(
            "assignments/resources/weather_raw.json",
            "r",
            encoding="utf-8"
        ) as f:
            weather_json = json.load(f)
    # Reshape hourly data
    hourly = weather_json["hourly"]
    weather_records = [
        {
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i]
        }
        for i in range(len(hourly["time"]))
    ]
    print(f"Loaded {len(weather_records)} weather records.")
    return weather_records

# Step 2: Transform
def classify_weather(records):

    """
    Classify first 24 hourly records as good, marginal, or bad for running.
    """

    client = OpenAI()
    classified_records = []
    for i, record in enumerate(records[:24], start=1):
        user_message = (
            f"Temperature: {record['temperature_2m']}C, "
            f"Precipitation: {record['precipitation']}mm"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                   "role": "user",
                    "content": user_message
                }
            ],
            temperature=0
        )
        label = response.choices[0].message.content.strip().lower()
        if label not in ["good", "marginal", "bad"]:
            label = "unknown"
        classified_records.append({
            **record,
            "conditions": label
        })
        if i % 6 == 0:
            print(f"Processed {i} records...")

    return classified_records

def save_results(classified_records):

    """
    Save classified results to JSON.
    """

    output_path = "assignments_10/outputs/weather_classified.json"
    with open(output_path, "w") as f:
        json.dump(classified_records, f, indent=2)
    print(f"Saved {len(classified_records)} records to {output_path}")
    
# Step 3: Write
def upload_classified_data(container_client, classified_records):

    """
    Upload classified weather records to Blob Storage.
    """
    today = date.today().isoformat()
    blob_path = f"processed/{today}/weather_classified.json"
    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(
        json.dumps(classified_records, indent=2),
        overwrite=True
    )

    print(f"Uploaded {blob_path}")
    print("Blob exists:", blob_client.exists())


# Step 4: Spot-Check
def spot_check(container_client, blob_path):
    """
    Download processed file and perform spot check.
    """
    blob_client = container_client.get_blob_client(blob_path)
    data = blob_client.download_blob().readall()
    records = json.loads(data.decode("utf-8"))
    df = pd.DataFrame(records)
    print("\nCondition Counts:")
    print(df["conditions"].value_counts())
    print("\nFirst 5 Rows:")
    print(df.head())
    return df
# Step 5: Save Output
def save_first_10_records(classified_records):

    """
    Save first 10 enriched records for mentor review.
    """

    with open("assignments_10/outputs/first_10_records.json", "w", encoding="utf-8") as f:
        json.dump(classified_records[:10], f, indent=2)

    print("Saved outputs/first_10_records.json")




def main():

    blob_path = f"raw/{date.today().isoformat()}/weather.json"
    weather_records = read_back(
        container_client, blob_path
    )

    classified_records = classify_weather(
        weather_records
    )
  
    upload_classified_data(
        container_client,
        classified_records
    )

    processed_blob_path = f"processed/{date.today().isoformat()}/weather_classified.json"

    spot_check(
        container_client,
        processed_blob_path
    )

    save_results(classified_records)

     # Step 5

    save_first_10_records(classified_records)


if __name__ == "__main__":
    main()