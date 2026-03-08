# Makefile — Orquestador del pipeline del Detector de Intrusiones
# Uso: make <target>
# Ejemplo: make test

.PHONY: install install-dev test lint pipeline clean help

# Variables
PYTHON = python
PIP = pip
PYTEST = pytest
RUFF = ruff

## help: Muestra esta lista de targets disponibles
help:
	@echo "Targets disponibles:"
	@echo "  install      Instalar dependencias de producción"
	@echo "  install-dev  Instalar dependencias de producción + desarrollo"
	@echo "  test         Ejecutar todos los tests con pytest"
	@echo "  lint         Ejecutar linter (ruff)"
	@echo "  lint-fix     Ejecutar linter con auto-fix"
	@echo "  pipeline     Ejecutar el pipeline completo (notebooks 00-03)"
	@echo "  clean        Limpiar archivos temporales y caches"

## install: Instalar el paquete con dependencias de producción
install:
	$(PIP) install -e .

## install-dev: Instalar con dependencias de desarrollo (pytest, ruff)
install-dev:
	$(PIP) install -e ".[dev]"

## test: Ejecutar todos los tests
test:
	$(PYTEST) tests/ -v

## lint: Verificar estilo de código con ruff
lint:
	$(RUFF) check src/ tests/

## lint-fix: Auto-corregir errores de estilo
lint-fix:
	$(RUFF) check --fix src/ tests/

## pipeline: Ejecutar el pipeline completo de notebooks en orden
pipeline:
	@echo "=== Ejecutando Pipeline Completo ==="
	@echo "--- Paso 00: Ingesta y Optimización ---"
	jupyter nbconvert --to notebook --execute notebooks/00_data_ingestion_and_optimization.ipynb --output 00_data_ingestion_and_optimization.ipynb
	@echo "--- Paso 01: Análisis Exploratorio ---"
	jupyter nbconvert --to notebook --execute notebooks/01_exploratory_data_analysis.ipynb --output 01_exploratory_data_analysis.ipynb
	@echo "--- Paso 02: Feature Engineering ---"
	jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering_and_preprocessing.ipynb --output 02_feature_engineering_and_preprocessing.ipynb
	@echo "--- Paso 03: Entrenamiento y Evaluación ---"
	jupyter nbconvert --to notebook --execute notebooks/03_model_training_and_evaluation.ipynb --output 03_model_training_and_evaluation.ipynb
	@echo "=== Pipeline Completado ==="

## clean: Limpiar archivos temporales
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "Archivos temporales eliminados."
