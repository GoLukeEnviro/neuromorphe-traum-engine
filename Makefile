# Neuromorphe Traum-Engine v2.0 - Makefile
# Vereinfachte Befehle für Entwicklung und Deployment

.PHONY: help install dev prod test clean setup docker-build docker-dev docker-prod docker-clean

# Default target
help:
	@echo "Neuromorphe Traum-Engine v2.0 - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup       - Setup database and directories"
	@echo "  make dev         - Start development server"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run code linting"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean temporary files"
	@echo ""
	@echo "Production:"
	@echo "  make prod        - Start production server"
	@echo "  make backup      - Backup database"
	@echo "  make restore     - Restore database"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    - Build Docker images"
	@echo "  make docker-dev      - Start development with Docker"
	@echo "  make docker-prod     - Start production with Docker"
	@echo "  make docker-test     - Run tests in Docker"
	@echo "  make docker-clean    - Clean Docker resources"
	@echo ""
	@echo "Audio Processing:"
	@echo "  make preprocess INPUT=<dir>  - Preprocess audio files"
	@echo "  make generate PROMPT=<text>  - Generate track from prompt"
	@echo ""
	@echo "Monitoring:"
	@echo "  make health      - Check system health"
	@echo "  make logs        - Show application logs"
	@echo "  make stats       - Show system statistics"

# Python Environment
PYTHON := python
PIP := pip
PYTHONPATH := src

# Directories
SRC_DIR := src
TEST_DIR := tests
DATA_DIR := data
LOGS_DIR := logs

# Installation
install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed successfully"

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio pytest-cov black flake8 mypy jupyter
	@echo "Development dependencies installed successfully"

# Setup
setup:
	@echo "Setting up Neuromorphe Traum-Engine..."
	@mkdir -p $(DATA_DIR)/audio_input
	@mkdir -p $(DATA_DIR)/audio_output
	@mkdir -p $(DATA_DIR)/processed_database
	@mkdir -p $(DATA_DIR)/generated_tracks
	@mkdir -p $(DATA_DIR)/database
	@mkdir -p $(LOGS_DIR)
	@echo "Directories created"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) run.py --mode setup
	@echo "Setup completed successfully"

setup-force:
	@echo "Force setup (will reset database)..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) run.py --mode setup --force

# Development
dev:
	@echo "Starting development server..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) run.py --mode dev --log-level DEBUG --verbose

dev-port:
	@echo "Starting development server on custom port..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) run.py --mode dev --port $(PORT) --log-level DEBUG

# Production
prod:
	@echo "Starting production server..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) run.py --mode prod --workers 4

prod-port:
	@echo "Starting production server on custom port..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) run.py --mode prod --port $(PORT) --workers $(WORKERS)

# Testing
test:
	@echo "Running system tests..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) run.py --mode test

test-unit:
	@echo "Running unit tests..."
	PYTHONPATH=$(PYTHONPATH) pytest $(TEST_DIR) -v

test-coverage:
	@echo "Running tests with coverage..."
	PYTHONPATH=$(PYTHONPATH) pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html

test-integration:
	@echo "Running integration tests..."
	PYTHONPATH=$(PYTHONPATH) pytest $(TEST_DIR)/integration -v

# Code Quality
lint:
	@echo "Running code linting..."
	flake8 $(SRC_DIR) --max-line-length=100 --ignore=E203,W503
	mypy $(SRC_DIR) --ignore-missing-imports

format:
	@echo "Formatting code..."
	black $(SRC_DIR) --line-length=100
	black $(TEST_DIR) --line-length=100

format-check:
	@echo "Checking code formatting..."
	black $(SRC_DIR) --check --line-length=100
	black $(TEST_DIR) --check --line-length=100

# Audio Processing
preprocess:
	@echo "Preprocessing audio files from $(INPUT)..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) run.py --mode preprocess --input $(INPUT)

generate:
	@echo "Generating track from prompt: $(PROMPT)"
	@echo "Note: Use API endpoint for track generation"
	curl -X POST "http://localhost:8000/api/v1/tracks/generate" \
		-H "Content-Type: application/json" \
		-d '{"prompt": "$(PROMPT)", "duration": 180}'

# Monitoring
health:
	@echo "Checking system health..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) run.py --mode health

health-api:
	@echo "Checking API health..."
	curl -s http://localhost:8000/health | python -m json.tool

stats:
	@echo "Showing system statistics..."
	curl -s http://localhost:8000/api/v1/system/stats | python -m json.tool

logs:
	@echo "Showing recent logs..."
	tail -f $(LOGS_DIR)/neuromorphe_engine.log

logs-error:
	@echo "Showing error logs..."
	tail -f $(LOGS_DIR)/error.log

# Database
backup:
	@echo "Creating database backup..."
	@mkdir -p backups
	cp $(DATA_DIR)/database/neuromorphe_engine.db backups/backup_$(shell date +%Y%m%d_%H%M%S).db
	@echo "Backup created in backups/"

restore:
	@echo "Restoring database from $(BACKUP)..."
	cp $(BACKUP) $(DATA_DIR)/database/neuromorphe_engine.db
	@echo "Database restored"

vacuum:
	@echo "Optimizing database..."
	sqlite3 $(DATA_DIR)/database/neuromorphe_engine.db "VACUUM; ANALYZE;"
	@echo "Database optimized"

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	@echo "Cleanup completed"

clean-data:
	@echo "WARNING: This will delete all data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(DATA_DIR)/*; \
		rm -rf $(LOGS_DIR)/*; \
		echo "Data cleaned"; \
	else \
		echo "Cancelled"; \
	fi

# Docker Commands
docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-build-dev:
	@echo "Building development Docker image..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml build

docker-dev:
	@echo "Starting development environment with Docker..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

docker-dev-bg:
	@echo "Starting development environment in background..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

docker-prod:
	@echo "Starting production environment with Docker..."
	docker-compose up -d

docker-prod-full:
	@echo "Starting full production stack (with PostgreSQL, Redis, Monitoring)..."
	docker-compose --profile postgres --profile redis --profile monitoring up -d

docker-test:
	@echo "Running tests in Docker..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm test-runner

docker-shell:
	@echo "Opening shell in development container..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm dev-tools bash

docker-logs:
	@echo "Showing Docker logs..."
	docker-compose logs -f neuromorphe-engine

docker-stop:
	@echo "Stopping Docker containers..."
	docker-compose down

docker-clean:
	@echo "Cleaning Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

# Jupyter
jupyter:
	@echo "Starting Jupyter Lab..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile jupyter up jupyter

# Monitoring
monitoring:
	@echo "Starting monitoring stack..."
	docker-compose --profile monitoring up -d prometheus grafana
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"

# API Testing
api-test:
	@echo "Testing API endpoints..."
	@echo "Health check:"
	curl -s http://localhost:8000/health
	@echo "\nSystem stats:"
	curl -s http://localhost:8000/api/v1/system/stats
	@echo "\nStems list:"
	curl -s http://localhost:8000/api/v1/stems/

# Performance Testing
perf-test:
	@echo "Running performance tests..."
	@echo "Note: Install 'ab' (Apache Bench) for load testing"
	ab -n 100 -c 10 http://localhost:8000/health

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "API documentation available at: http://localhost:8000/docs"
	@echo "ReDoc documentation available at: http://localhost:8000/redoc"

# Quick Start
quickstart: install setup
	@echo "Quick start completed!"
	@echo "Run 'make dev' to start the development server"
	@echo "Or 'make docker-dev' to use Docker"

# CI/CD
ci: install-dev lint format-check test-coverage
	@echo "CI pipeline completed"

# Release
release:
	@echo "Preparing release..."
	@echo "Current version: $(shell grep __version__ src/__init__.py)"
	@echo "Run tests, update version, and tag release"
	make ci
	@echo "Ready for release!"

# Environment Info
info:
	@echo "System Information:"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Working Directory: $(shell pwd)"
	@echo "Python Path: $(PYTHONPATH)"
	@echo "Data Directory: $(DATA_DIR)"
	@echo "Logs Directory: $(LOGS_DIR)"