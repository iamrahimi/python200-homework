# =========================
# Prefect Question 1
# =========================
"""
A @task in Prefect is used for individual units of work, while a @flow is the overall pipeline that coordinates tasks and defines execution order.
A simple helper function like converting Celsius to Fahrenheit is a pure in-memory calculation with no I/O, so it does not need to be a @task. It would add unnecessary overhead unless it is part of a larger tracked pipeline where observability is important.
"""


# =========================
# Prefect Question 2
# =========================
# Decorator line only:
"""
@task(retries=3, retry_delay_seconds=30)
"""


# =========================
# Prefect Question 3
# =========================
"""
To understand why transform failed, I would go to the Prefect UI and click on the failed 'transform' task run.
There I would check the logs, error message, and stack trace to see exactly what caused the failure. I would also look at input/output data for that task.
Since 'load' never ran, I would confirm that it was skipped due to the failure dependency from 'transform'.
"""


# =========================
# Production Question 1
# =========================
"""
raise_for_status() checks the HTTP response and automatically raises an exception if the status code is 4xx or 5xx.
This is better than manually printing errors because it properly fails the task and stops downstream execution in a Prefect pipeline.
If an API returns a 500 error, raise_for_status() will fail the task and prevent downstream tasks from running. If you only print an error, the pipeline may continue running with invalid or empty data.
"""


# =========================
# Production Question 2
# =========================
"""
Using overwrite=True ensures that the corrected run replaces the previous partial or corrupted output file.
Without overwrite=True, the pipeline might fail to update the existing blob or could leave behind outdated or incomplete data from a failed run.
With overwrite=True, re-running the pipeline produces a clean final output that reflects the latest successful execution.
"""


# =========================
# Production Question 3
# =========================
from prefect import task, get_run_logger

@task
def log_loaded_records(records: list, blob_path: str):
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records from {blob_path}")