# ---- Build Stage ----
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build

# Set up environment
ENV LANG=C.UTF-8
ENV DOTNET_ROOT=/usr/share/dotnet

# Set working directory
WORKDIR /src

# Copy .NET source (passed in Makefile via context)
COPY ./docker/mercury/*.csproj ./mercury/

WORKDIR /src/mercury
RUN dotnet restore

# Copy everything except bin/obj (via .dockerignore)
WORKDIR /src
COPY ./docker/mercury ./mercury

# Publish to output path for runtime container
RUN dotnet publish ./mercury -c Release -o /home/site/wwwroot/mercury


# ---- Runtime Stage ----
FROM mcr.microsoft.com/azure-functions/python:4-python3.11

# Install .NET Core Runtime
RUN apt-get update && \
    apt-get install -y wget apt-transport-https software-properties-common gnupg && \
    wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y aspnetcore-runtime-8.0 && \
    rm -rf /var/lib/apt/lists/*

# Copy pre-built .NET app
WORKDIR /home/site/wwwroot/mercury
COPY --from=build /home/site/wwwroot/mercury .

# Copy Python app & shared directories
WORKDIR /home/site/wwwroot

# Function folders
COPY AlphaVee/ ./AlphaVee/
COPY DirectIndexing/ ./DirectIndexing/
COPY KeebeckMultiStrategy/ ./KeebeckMultiStrategy/
COPY MasterSystemLive/ ./MasterSystemLive/
COPY NasdaqDorseyWright/ ./NasdaqDorseyWright/
COPY NDWRebalanceStopLoss/ ./NDWRebalanceStopLoss/
COPY PassiveIndex/ ./PassiveIndex/

# Python logic and shared folders
COPY utils/ ./utils/
COPY stats/ ./stats/

# Core config and function entry point
COPY function_app.py ./
COPY host.json ./
COPY local.settings.json ./
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set environment variables
ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true \
    FUNCTIONS_WORKER_RUNTIME=python \
    PYTHONNET_RUNTIME=coreclr

# ENV AzureWebJobsStorage=UseDevelopmentStorage=true \
#     Storage=UseDevelopmentStorage=true \
#     FUNCTIONS_WORKER_RUNTIME=python


# # Default CMD
# CMD ["python", "-m", "azure_functions_worker"]

