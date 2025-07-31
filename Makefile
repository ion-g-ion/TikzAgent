# TikZ Agent Docker Makefile

# Default values
IMAGE_NAME ?= tikz-agent
PORT ?= 8501
CONTAINER_NAME ?= tikz-agent-demo
APP_URL ?=

.PHONY: help build run run-detached stop logs clean shell test-build

help:	## Show this help message
	@echo "TikZ Agent Docker Commands:"
	@echo "=========================="
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  %-15s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

build:	## Build the Docker image
	@echo "🔨 Building Docker image..."
	docker build -t $(IMAGE_NAME) .

run:		## Run the container interactively
	@echo "🚀 Starting TikZ Agent on port $(PORT)..."
	@mkdir -p output
	docker run --rm -it \
		-p $(PORT):$(PORT) \
		-e PORT=$(PORT) \
		-e APP_URL="$(APP_URL)" \
		-e OPENAI_API_KEY="$(OPENAI_API_KEY)" \
		-e ANTHROPIC_API_KEY="$(ANTHROPIC_API_KEY)" \
		-e GOOGLE_API_KEY="$(GOOGLE_API_KEY)" \
		-e OPENROUTER_API_KEY="$(OPENROUTER_API_KEY)" \
		-v $(PWD)/output:/app/output \
		--name $(CONTAINER_NAME)-temp \
		$(IMAGE_NAME)

run-detached:	## Run the container in detached mode
	@echo "🚀 Starting TikZ Agent in background on port $(PORT)..."
	@mkdir -p output
	docker run -d \
		--restart unless-stopped \
		-p $(PORT):$(PORT) \
		-e PORT=$(PORT) \
		-e APP_URL="$(APP_URL)" \
		-e OPENAI_API_KEY="$(OPENAI_API_KEY)" \
		-e ANTHROPIC_API_KEY="$(ANTHROPIC_API_KEY)" \
		-e GOOGLE_API_KEY="$(GOOGLE_API_KEY)" \
		-e OPENROUTER_API_KEY="$(OPENROUTER_API_KEY)" \
		-v $(PWD)/output:/app/output \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME)
	@echo "✅ Container started as '$(CONTAINER_NAME)'"
	@echo "   Visit http://localhost:$(PORT)"

stop:		## Stop the detached container
	@echo "🛑 Stopping container..."
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

logs:		## Show container logs
	docker logs -f $(CONTAINER_NAME)

shell:	## Open a shell in the running container
	docker exec -it $(CONTAINER_NAME) /bin/bash

clean:	## Remove the Docker image and container
	@echo "🧹 Cleaning up..."
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	docker rmi $(IMAGE_NAME) || true

test-build:	## Test build without cache
	@echo "🧪 Testing clean build..."
	docker build --no-cache -t $(IMAGE_NAME)-test .
	@echo "✅ Build test successful"
	docker rmi $(IMAGE_NAME)-test

# Docker Compose commands
compose-up:	## Start with docker-compose
	docker-compose up --build

compose-down:	## Stop docker-compose services
	docker-compose down

compose-logs:	## Show docker-compose logs
	docker-compose logs -f

# Environment check
check-env:	## Check environment variables
	@echo "Environment Variables:"
	@echo "====================="
	@echo "PORT: $(PORT)"
	@echo "IMAGE_NAME: $(IMAGE_NAME)"
	@echo "CONTAINER_NAME: $(CONTAINER_NAME)"
	@echo "APP_URL: $(APP_URL)"
	@echo ""
	@echo "API Keys (set = ✅, not set = ❌):"
	@if [ -n "$(OPENAI_API_KEY)" ]; then echo "OPENAI_API_KEY: ✅"; else echo "OPENAI_API_KEY: ❌"; fi
	@if [ -n "$(ANTHROPIC_API_KEY)" ]; then echo "ANTHROPIC_API_KEY: ✅"; else echo "ANTHROPIC_API_KEY: ❌"; fi
	@if [ -n "$(GOOGLE_API_KEY)" ]; then echo "GOOGLE_API_KEY: ✅"; else echo "GOOGLE_API_KEY: ❌"; fi
	@if [ -n "$(OPENROUTER_API_KEY)" ]; then echo "OPENROUTER_API_KEY: ✅"; else echo "OPENROUTER_API_KEY: ❌"; fi 