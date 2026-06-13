Cloud Concepts Question 1
A  ==> The core economic model of cloud computing is pay-as-you-go, meaning you only pay for the computing resources you actually use instead of buying everything upfront.

It is different from owning your own servers because with cloud you don’t need to invest in hardware, maintenance, or setup—you just rent what you need. But with your own servers, you pay a large upfront cost and you are responsible for everything, even if you don’t fully use the system.
 


Cloud Concepts Question 2
A ==> Vertical scaling means upgrading one machine to make it stronger (more CPU, RAM, or GPU).
Horizontal scaling means adding more machines to share the workload.

You would use vertical scaling when a single task needs more power on one system, like speeding up a machine learning training job. You would use horizontal scaling when you need to handle more users or more tasks by distributing the work across multiple machines.

Q:- A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.
A:- This is horizontal scaling because the system needs to handle many more users, so the load can be spread across multiple servers.

Q:- A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.
A:-  This is vertical scaling because the user needs a single machine with more powerful hardware like a better GPU and more memory.

Q:- A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.
A:- This is horizontal scaling because the work can be split across many machines to process more files in parallel.




Cloud Concepts Question 3
A:- Classification

 Gmail -> SaaS because it is a ready-to-use application where you only use email without managing any infrastructure.
 Azure Virtual Machines -> IaaS because it provides raw virtual servers and you manage the OS, software, and setup yourself.
 Azure App Service -> PaaS because you just deploy your app while the platform handles servers, scaling, and runtime.
 AWS S3 (Simple Storage Service) -> IaaS because it gives storage infrastructure where you manage how data is organized and used, but not the hardware.
 GitHub Codespaces -> PaaS because it gives you a ready coding environment without needing to set up your own machine.
 Snowflake -> SaaS because it is a fully managed data platform you just use without handling infrastructure.

⸻

In simple words

IaaS (Infrastructure as a Service)
You get virtual machines and infrastructure, but you manage everything above it (OS, apps, setup).
Example: Azure Virtual Machines -> You manage installing software, configuring the system, and maintaining it.

PaaS (Platform as a Service)
You just deploy your application, and the platform handles servers and scaling.
Example: Azure App Service -> You only manage your code and deployment, not the underlying servers.

SaaS (Software as a Service)
You simply use the finished application. Nothing to manage.
Example: Gmail -> You just use email; everything else is handled for you.

 


Cloud Concepts Question 4
A:- A managed data platform like Snowflake or Databricks is a ready-made system for storing and analyzing data that runs on top of cloud infrastructure, so you don’t have to build everything yourself.

Compared to using a cloud provider like Azure directly, you gain speed, simplicity, and built-in data tools, but you give up some control, flexibility, and sometimes pay more for convenience.

 Cloud Concepts Question 5
A:- The cloud is usually not the best choice in two main situations:

1. When you need very low latency or real-time control, like systems that must respond instantly (e.g., industrial machines or high-frequency trading).

2. When you have a steady, long-term workload that is cheaper to run on your own hardware, where owning servers is more cost-efficient than renting cloud resources.


 Azure Basics 
 Azure Basics Question 1
    Azure Subscription: The top-level billing and management container for Azure resources. It controls access, quotas, and costs.
    Resource Group: A logical container inside a subscription that holds related Azure resources (VMs, databases, storage accounts, etc.) 


    Your Azure subscription is assigned to you and is yours to use.
    The resource group is shared by CTD and is used to organize resources for a project or team.
  

Azure Basics Question 2
  
    By default, Azure Cloud Shell is ephemeral, which means any files, settings, or changes you make are temporary and can be lost when the session ends. 

Azure Basics Question 3 
    SSH Private Key: A secret key that stays on your computer and should never be shared.
    SSH Public Key: A key that can be shared openly.

    The public key is uploaded to the remote system you want to connect to. This is safe because the public key cannot be used to determine or recreate the private key. The remote system uses the public key to verify that you possess the matching private key, without ever needing to know the private key itself.


Azure Basics Question 4
    {
    "environmentName": "AzureCloud",
    "homeTenantId": "REDACTED",
    "id": "REDACTED",
    "isDefault": true,
    "managedByTenants": [],
    "name": "CTD Nonprofit Sponsorship",
    "state": "Enabled",
    "tenantId": "REDACTED",
    "user": {
        "cloudShellID": true,
        "name": "REDACTED",
        "type": "user"
    }
    }

    The output shows that I am signed in to the Azure subscription “CTD Nonprofit Sponsorship”, which is my default and active subscription. When I add --output table, Azure displays the same information in a more readable table format instead of JSON.
 
