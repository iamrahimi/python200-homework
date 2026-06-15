
import requests
import json
from datetime import date
import os
import pandas as pd

# Video link https://youtu.be/VY-twBiwHYY

# ----------------------------
# Setup constants
# ----------------------------

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

# Step 1: Extract

def extract_weather(lat=35.2271, lon=-80.8431):
    """
        Extract 7 days of hourly weather data from Open-Meteo API.
        Default location: Charlotte, NC
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=7"
    )
    response = requests.get(url)
    response.raise_for_status()  # stops program if API fails
    return response.json()


# Step 2: Serialize

def serialize_weather(weather_data):

    """
        Convert API response (dict) into UTF-8 encoded JSON bytes.
    """
    json_string = json.dumps(weather_data)
    json_bytes = json_string.encode("utf-8")
    return json_bytes

# Step 3: Load (Upload to Azure Blob Storage)


def load_to_blob(container_client, data_bytes):

    """
        Upload serialized weather data to Azure Blob Storage.
    """
    today = date.today().isoformat()
    blob_path = f"raw/{today}/weather.json"
    container_client.upload_blob(
        name=blob_path,
        data=data_bytes,
        overwrite=True
    )
    print(f"Uploaded to: {blob_path}")
    print(f"Bytes uploaded: {len(data_bytes)}")


# Step 4: Verify (List blobs in container)

def verify_upload(container_client):

    """
        List all blobs in the container and print name + size.
    """
    blobs = container_client.list_blobs()
    for blob in blobs:
        print(f"{blob.name}: {blob.size} bytes")

#  Step 5: Read Back (Download + Parse + DataFrame)


def read_back(container_client, blob_path):

    """
        Download blob, save locally, parse JSON, and load hourly data into DataFrame.
    """
    # -------------------------
    # Step 1: Download blob
    # -------------------------
    blob_client = container_client.get_blob_client(blob_path)
    download_data = blob_client.download_blob().readall()
    # -------------------------
    # Step 2: Save raw JSON locally
    # -------------------------
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/weather_raw.json", "wb") as f:
        f.write(download_data)
    # -------------------------
    # Step 3: Parse JSON
    # -------------------------
    weather_json = json.loads(download_data.decode("utf-8"))
    # -------------------------
    # Step 4: Extract hourly data
    # -------------------------
    hourly_data = weather_json["hourly"]
    df = pd.DataFrame(hourly_data)
    # -------------------------
    # Step 5: Show results
    # -------------------------
    print(df.head())
    return df


if __name__ == "__main__":
    data = extract_weather()
    bytes_data = serialize_weather(data)
    load_to_blob(container_client, bytes_data)
    verify_upload(container_client)
    read_back(container_client, f"raw/{date.today().isoformat()}/weather.json")
    print("Pipeline finished successfully")


