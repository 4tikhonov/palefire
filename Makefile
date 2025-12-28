# Pale Fire - Makefile
# Convenience commands for Docker and development

.PHONY: help build up down restart logs ps clean test shell cli

# Default target
.DEFAULT_GOAL := help

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Pale Fire - Docker Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Docker Compose Commands
build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose build

up: ## Start all services
	@echo "$(BLUE)Starting all services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo "API: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"
	@echo "Neo4j: http://localhost:7474"

down: ## Stop all services
	@echo "$(YELLOW)Stopping all services...$(NC)"
	docker-compose down

restart: ## Restart all services
	@echo "$(BLUE)Restarting all services...$(NC)"
	docker-compose restart

logs: ## View logs (all services)
	docker-compose logs -f

logs-api: ## View API logs
	docker-compose logs -f palefire-api

logs-neo4j: ## View Neo4j logs
	docker-compose logs -f neo4j

logs-ollama: ## View Ollama logs
	docker-compose logs -f ollama

ps: ## Show service status
	docker-compose ps

# Setup Commands
setup: ## Initial setup (pull models, etc.)
	@echo "$(BLUE)Running initial setup...$(NC)"
	@echo "Waiting for services to be healthy..."
	@sleep 30
	@echo "Pulling Ollama models..."
	docker-compose exec ollama ollama pull deepseek-r1:7b
	docker-compose exec ollama ollama pull nomic-embed-text
	@echo "$(GREEN)Setup complete!$(NC)"

models: ## Pull Ollama models
	@echo "$(BLUE)Pulling Ollama models...$(NC)"
	docker-compose exec ollama ollama pull deepseek-r1:7b
	docker-compose exec ollama ollama pull nomic-embed-text
	@echo "$(GREEN)Models pulled!$(NC)"

# CLI Commands
cli: ## Start CLI container
	docker-compose --profile cli up -d palefire-cli
	@echo "$(GREEN)CLI container started!$(NC)"
	@echo "Run: make shell-cli"

shell: ## Open shell in API container
	docker-compose exec palefire-api bash

shell-cli: ## Open shell in CLI container
	docker-compose exec palefire-cli bash

shell-neo4j: ## Open Cypher shell in Neo4j
	docker-compose exec neo4j cypher-shell -u neo4j -p palefire123

# Data Commands
ingest-demo: ## Ingest demo data
	@echo "$(BLUE)Ingesting demo data...$(NC)"
	docker-compose exec palefire-cli python palefire-cli.py ingest --demo
	@echo "$(GREEN)Demo data ingested!$(NC)"

query: ## Run a test query
	@echo "$(BLUE)Running test query...$(NC)"
	docker-compose exec palefire-cli python palefire-cli.py query "Who is Kamala Harris?"

config: ## Show configuration
	docker-compose exec palefire-cli python palefire-cli.py config

clean-db: ## Clean Neo4j database
	@echo "$(YELLOW)Cleaning database...$(NC)"
	docker-compose exec palefire-cli python palefire-cli.py clean --confirm
	@echo "$(GREEN)Database cleaned!$(NC)"

# Testing Commands
test: ## Run tests
	docker-compose exec palefire-api pytest

test-cov: ## Run tests with coverage
	docker-compose exec palefire-api pytest --cov=. --cov-report=html
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(NC)"

verify: ## Verify test suite
	docker-compose exec palefire-api ./verify_tests.sh

# Maintenance Commands
clean: ## Stop and remove containers, networks, volumes
	@echo "$(YELLOW)Cleaning up Docker resources...$(NC)"
	docker-compose down -v
	@echo "$(GREEN)Cleanup complete!$(NC)"

prune: ## Remove unused Docker resources
	@echo "$(YELLOW)Pruning Docker resources...$(NC)"
	docker system prune -f
	@echo "$(GREEN)Prune complete!$(NC)"

backup: ## Backup Neo4j data
	@echo "$(BLUE)Backing up Neo4j data...$(NC)"
	@mkdir -p backups
	docker run --rm \
		-v palefire_neo4j_data:/data \
		-v $(PWD)/backups:/backup \
		alpine tar czf /backup/neo4j-backup-$$(date +%Y%m%d_%H%M%S).tar.gz /data
	@echo "$(GREEN)Backup complete!$(NC)"

restore: ## Restore Neo4j data (requires BACKUP_FILE variable)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "$(YELLOW)Usage: make restore BACKUP_FILE=backups/neo4j-backup-XXXXXX.tar.gz$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Restoring Neo4j data from $(BACKUP_FILE)...$(NC)"
	docker run --rm \
		-v palefire_neo4j_data:/data \
		-v $(PWD)/backups:/backup \
		alpine tar xzf /backup/$$(basename $(BACKUP_FILE)) -C /
	@echo "$(GREEN)Restore complete!$(NC)"

# Health Checks
health: ## Check service health
	@echo "$(BLUE)Checking service health...$(NC)"
	@curl -s http://localhost:8000/health | jq . || echo "API not responding"
	@curl -s http://localhost:7474 > /dev/null && echo "$(GREEN)Neo4j: OK$(NC)" || echo "$(YELLOW)Neo4j: DOWN$(NC)"
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "$(GREEN)Ollama: OK$(NC)" || echo "$(YELLOW)Ollama: DOWN$(NC)"

# Development Commands
dev: ## Start in development mode (with hot reload)
	@echo "$(BLUE)Starting in development mode...$(NC)"
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

rebuild: ## Rebuild and restart services
	@echo "$(BLUE)Rebuilding and restarting...$(NC)"
	docker-compose build --no-cache
	docker-compose up -d
	@echo "$(GREEN)Rebuild complete!$(NC)"

# Quick Start
quickstart: build up setup ingest-demo ## Complete quick start (build, start, setup, ingest demo)
	@echo "$(GREEN)Quick start complete!$(NC)"
	@echo "Try: make query"

# Monitoring
stats: ## Show resource usage
	docker stats --no-stream

top: ## Show running processes
	docker-compose top

# Documentation
docs: ## Open documentation in browser
	@echo "Opening documentation..."
	@open http://localhost:8000/docs || xdg-open http://localhost:8000/docs || echo "Open http://localhost:8000/docs in your browser"

