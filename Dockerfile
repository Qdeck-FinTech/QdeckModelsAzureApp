# To enable ssh & remote debugging on app service change the base image to the one below
# FROM mcr.microsoft.com/azure-functions/python:4-python3.11-appservice
FROM mcr.microsoft.com/azure-functions/python:4-python3.11

# # See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true

ENV AzureWebJobsStorage="UseDevelopmentStorage=true"
ENV FUNCTIONS_WORKER_RUNTIME="python"

# Environment variable for .NET
ENV DOTNET_ROOT=/usr/share/dotnet

ENV WEBSITE_HOSTNAME=localhost:80

# Install prerequisites
RUN apt-get update && \
    apt-get install -y wget apt-transport-https software-properties-common gnupg && \
    wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb

# Install .NET 8 SDK
RUN apt-get update && \
    apt-get install -y dotnet-sdk-8.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libkrb5-dev \
    libssl-dev \
    libpam0g-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Confirm .NET installation
RUN dotnet --info

WORKDIR /home/site/wwwroot

# Copy models directories
COPY AlphaVee/ ./AlphaVee/ 
COPY MasterSystemLive/ ./MasterSystemLive/ 
COPY NasdaqDorseyWright/ ./NasdaqDorseyWright/ 
COPY PassiveIndex/ ./PassiveIndex/ 
COPY KeebeckMultiStrategy/ ./KeebeckMultiStrategy/ 
COPY DirectIndexing/ ./DirectIndexing/ 

# Copy bin and utils directories
COPY bin/ ./bin/
COPY utils/ ./utils/
COPY .dockerignore ./

# Copy main application files
COPY function_app.py ./
COPY host.json ./
COPY local.settings.json ./

# Install Python dependencies
COPY requirements.txt ./
RUN pip install -r ./requirements.txt

