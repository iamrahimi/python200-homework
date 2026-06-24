

Part 2: Cost Analysis

Task 01: https://youtu.be/TvkP2a7Nzqc

Task 02: https://youtu.be/Vi2fffmEv94

Azure Cost Estimation Exercise

Scenario A

VM: Standard_B1s
Hours: 160/month
Region: East US
Estimated Cost: $1.66per month
Summary: Low-cost development workload

Scenario B

VM: Standard_NC6s_v3 (GPU)
Hours: 730/month
SQL Database: 4 vCores
Blob Storage: 1 TB
Estimated Cost: $1511 per month
Summary: High-cost production analytics system

Observations

Compute (especially GPU) is the most expensive part
Running 24/7 drastically increases cost
Storage is relatively cheap compared to compute
Small VM vs GPU VM cost difference is very large