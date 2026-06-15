

# Azure Authentication

# Q1

# When a Python script runs locally and uses DefaultAzureCredential,
# it can authenticate using credentials from the Azure CLI if you are
# signed in. Before running the script, you must run:
#
# az login
#
# This command signs you in to Azure and stores authentication tokens
# locally. DefaultAzureCredential checks several authentication methods
# in a specific order. When it finds a valid Azure CLI login session,
# it automatically uses those stored Azure CLI credentials to obtain
# access tokens for Azure services. This allows the script to
# authenticate without hardcoding usernames, passwords, or secrets.


# Q2

# A deployed pipeline running on an Azure VM, App Service, Container App,
# Kubernetes cluster, or other Azure-hosted service cannot rely on
# `az login` because there is no interactive user available to sign in.
# Production workloads must run automatically without requiring someone
# to enter credentials.
#
# Instead, Azure resources typically use a Managed Identity. Azure
# automatically creates and manages the identity, and the application
# can request access tokens from Azure without storing secrets,
# usernames, or passwords.
#
# The same Python code works without changes because
# DefaultAzureCredential checks multiple authentication methods in a
# predefined order. On a local machine, it may find Azure CLI
# credentials from `az login`. In Azure, it detects the Managed
# Identity endpoint and uses that identity instead.
#
# This allows developers to write code once:
#
#     credential = DefaultAzureCredential()
#
# and use it both locally and in Azure deployments without modifying
# the authentication code.


# Q3

# If a script creates a DefaultAzureCredential and immediately raises
# an AuthenticationError, the two most likely causes are:
#
# 1. No valid local authentication exists.
#    For example, you have not signed in with Azure CLI, your login has
#    expired, or you are using the wrong Azure account.
#
#    How to diagnose:
#    - Run: az account show
#    - If it returns an error or shows the wrong account, sign in again:
#          az login
#    - Verify that the correct subscription and tenant are selected.
#
# 2. The application does not have a valid identity or permissions.
#    For example, a Managed Identity is not enabled on the Azure
#    resource, or the identity does not have permission to access the
#    requested Azure service.
#
#    How to diagnose:
#    - Check whether Managed Identity is enabled on the Azure resource.
#    - Review the full error message for authorization or permission
#      failures.
#    - Verify that the identity has the required Azure RBAC role or
#      access permissions for the target resource.
#
# Additional troubleshooting:
# - Read the complete AuthenticationError message because
#   DefaultAzureCredential reports which credential types it attempted
#   and why each one failed.
# - Confirm that network connectivity to Azure services is available.


# Blob Storage

# Q1

# Azure Blob Storage has a three-level hierarchy:
#
# 1. Storage Account
#    - The top-level container that owns and manages all storage
#      resources. It provides a unique namespace and settings such as
#      security, networking, and access controls.
#
# 2. Container
#    - A logical grouping inside a storage account that organizes blobs.
#      Containers are similar to folders, although they are the primary
#      organizational unit in Blob Storage.
#
# 3. Blob
#    - The actual file or data object being stored, such as an image,
#      document, video, log file, or CSV file.
#
# Filing Cabinet Analogy:
#
# - Storage Account = Filing Cabinet
#   The entire cabinet that holds all of your documents.
#
# - Container = Drawer
#   Each drawer organizes a category of documents, such as Finance,
#   Human Resources, or Projects.
#
# - Blob = Individual File
#   A specific document stored inside a drawer, such as
#   "budget_2026.xlsx" or "employee_list.pdf".
#
# In this analogy, you first open the filing cabinet (storage account),
# then choose a drawer (container), and finally access a specific file (blob).


# Q2

# A. A REST API returns a JSON payload each hour. You need to store the
# raw responses for reprocessing later.
#
# I would use Blob Storage because it is designed for storing large
# amounts of unstructured data such as JSON files and is cost-effective
# for archival and reprocessing purposes.
# B. Your pipeline produces a table of 50 million customer transactions
# that your analytics team queries by date range and customer ID every day.
#
# I would use a relational database such as Azure SQL because the data
# is highly structured and requires efficient querying, filtering, and
# indexing by fields like date and customer ID.
# C. A computer vision model produces image embeddings as NumPy arrays.
# You need to save them between pipeline runs.
#
# I would use Blob Storage because NumPy arrays can be saved as files
# (such as .npy files), and Blob Storage is well suited for storing and
# retrieving large binary objects between pipeline executions.


# Q3

def list_container(container_client):
    """
        Print the name and size (in bytes) of every blob in the container.
    """
    for blob in container_client.list_blobs():
        print(f"{blob.name}: {blob.size} bytes")

# Q4

def upload_text(container_client, blob_name, text):
    """
        Encode a string as UTF-8 and upload it as a blob,
        overwriting any existing blob with the same name.
    """
    data = text.encode("utf-8")
    container_client.upload_blob(
        name=blob_name,
        data=data,
        overwrite=True
    )