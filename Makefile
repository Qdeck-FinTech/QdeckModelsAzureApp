include makefile_config.env

# Paths
SRC=../Mercury.NetCore/Mercury
OUT=./docker/mercury
DOCKERFILE=Dockerfile
FULL_NEW_IMAGE_NAME=$(LOGIN_SERVER)/$(BASE_IMAGE_NAME):$(TAG)

export DOCKER_DEFAULT_PLATFORM=linux/amd64

confirm_tag:
	@if [ -z "$(TAG)" ]; then \
		echo "Error: TAG was not defined. Use: make build TAG=v0.0.1"; \
		exit 1; \
	fi

prepare_dotnet:
	@echo "Copying .NET Core Mercury from $(SRC) to $(OUT)..."
	@rm -rf $(OUT)
	@mkdir -p $(OUT)
	@rsync -av --exclude='bin' --exclude='obj' $(SRC)/ $(OUT)/
	@echo "Copy complete."

build: confirm_tag prepare_dotnet
	@echo "Building Docker image: $(FULL_NEW_IMAGE_NAME)..."
	docker build --platform=linux/amd64 -f $(DOCKERFILE) -t $(FULL_NEW_IMAGE_NAME) .

run: confirm_tag
	@echo "Running Docker image: $(FULL_NEW_IMAGE_NAME)..."
	docker run -it -p 8080:80 --rm --env-file ../../../config.env $(FULL_NEW_IMAGE_NAME)

check_azure_image:
	@az acr login --name $(REGISTRY)
	@echo "Checking image currently used by $(APP_NAME)..."
	@az functionapp config container show --resource-group $(RESOURCE_GROUP) --name $(APP_NAME)

list_local_images:
	@echo "Local images with name containing '$(BASE_IMAGE_NAME)':"
	@docker images --format "{{.Repository}}:{{.Tag}}" | grep '$(BASE_IMAGE_NAME):' | sort -V

deploy: confirm_tag
	@echo "Deploying image to Azure Container Registry..."
	# az acr login --name $(REGISTRY)
	docker push $(FULL_NEW_IMAGE_NAME)
	az acr update -n $(REGISTRY) --admin-enabled true
	# az functionapp config container set \
	# 	--image $(FULL_NEW_IMAGE_NAME) \
	# 	--registry-password $(REGISTRY_PASSWORD) \
	# 	--registry-username $(REGISTRY_USER) \
	# 	--name $(APP_NAME) \
	# 	--resource-group $(RESOURCE_GROUP)

build_and_deploy: build deploy


run-local:
	@echo "Running local Azure Function + Azurite using Docker Compose..."
	docker-compose up --build

stop-local:
	@echo "Stopping and removing local containers..."
	docker-compose down

clean:
	rm -rf $(OUT)
	@echo "🧹 Cleaned output folder: $(OUT)"
