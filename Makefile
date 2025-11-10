.PHONY: help install verify setup-dataset clean-data train predict run test clean clean-all

# Variables
PYTHON := python
PIP := pip
VENV := venv

# Colores para output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Muestra esta ayuda
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║     🍎 Clasificador de Frutas CNN - Comandos Make 🍌        ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Comandos disponibles:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)Flujo de trabajo típico:$(NC)"
	@echo "  1. make install         (instalar dependencias)"
	@echo "  2. make verify          (verificar instalación)"
	@echo "  3. make setup-dataset   (configurar dataset)"
	@echo "  4. make clean-data      (limpiar datos)"
	@echo "  5. make train           (entrenar modelo)"
	@echo "  6. make run             (iniciar app web)"
	@echo ""

install: ## Instala todas las dependencias
	@echo "$(GREEN)📦 Instalando dependencias...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ Dependencias instaladas$(NC)"

install-dev: ## Instala dependencias de desarrollo
	@echo "$(GREEN)📦 Instalando dependencias de desarrollo...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install jupyter notebook ipython
	@echo "$(GREEN)✅ Dependencias de desarrollo instaladas$(NC)"

verify: ## Verifica la instalación
	@echo "$(BLUE)🔍 Verificando instalación...$(NC)"
	$(PYTHON) utils/verify_installation.py

setup-dataset: ## Ayuda a configurar el dataset
	@echo "$(BLUE)📊 Configurando dataset...$(NC)"
	$(PYTHON) utils/download_sample_dataset.py

clean-data: ## Limpia y prepara el dataset
	@echo "$(BLUE)🧹 Limpiando y preparando datos...$(NC)"
	$(PYTHON) scripts/data_preparation.py
	@echo "$(GREEN)✅ Datos preparados$(NC)"

train: ## Entrena el modelo CNN
	@echo "$(BLUE)🧠 Entrenando modelo...$(NC)"
	$(PYTHON) scripts/train_model.py
	@echo "$(GREEN)✅ Modelo entrenado$(NC)"

predict: ## Realiza predicción (requiere imagen como argumento: make predict IMG=imagen.jpg)
	@if [ -z "$(IMG)" ]; then \
		echo "$(RED)❌ Error: Especifica una imagen$(NC)"; \
		echo "$(YELLOW)Uso: make predict IMG=ruta/a/imagen.jpg$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)🔍 Realizando predicción...$(NC)"
	$(PYTHON) scripts/predict.py $(IMG) --all

run: ## Inicia la aplicación web
	@echo "$(BLUE)🚀 Iniciando aplicación web...$(NC)"
	@echo "$(GREEN)📱 Accede en: http://localhost:5000$(NC)"
	$(PYTHON) app.py

test: ## Ejecuta tests (si existen)
	@echo "$(BLUE)🧪 Ejecutando tests...$(NC)"
	@if [ -d "tests" ]; then \
		$(PYTHON) -m pytest tests/; \
	else \
		echo "$(YELLOW)⚠️  No se encontró carpeta de tests$(NC)"; \
	fi

config: ## Muestra la configuración actual
	@echo "$(BLUE)⚙️  Configuración del proyecto:$(NC)"
	$(PYTHON) config.py

clean: ## Limpia archivos temporales
	@echo "$(YELLOW)🧹 Limpiando archivos temporales...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf static/uploads/*
	@echo "$(GREEN)✅ Archivos temporales eliminados$(NC)"

clean-dataset: ## Elimina el dataset procesado
	@echo "$(YELLOW)🗑️  Eliminando dataset procesado...$(NC)"
	rm -rf dataset/processed
	@echo "$(GREEN)✅ Dataset procesado eliminado$(NC)"

clean-models: ## Elimina los modelos entrenados
	@echo "$(YELLOW)🗑️  Eliminando modelos...$(NC)"
	rm -rf models/*.h5
	rm -rf models/*.keras
	@echo "$(GREEN)✅ Modelos eliminados$(NC)"

clean-all: clean clean-dataset clean-models ## Limpia todo (temporales, dataset, modelos)
	@echo "$(GREEN)✅ Limpieza completa realizada$(NC)"

venv: ## Crea un entorno virtual
	@echo "$(BLUE)🐍 Creando entorno virtual...$(NC)"
	python3 -m venv $(VENV)
	@echo "$(GREEN)✅ Entorno virtual creado$(NC)"
	@echo "$(YELLOW)Actívalo con: source $(VENV)/bin/activate$(NC)"

freeze: ## Congela las dependencias actuales
	@echo "$(BLUE)📋 Congelando dependencias...$(NC)"
	$(PIP) freeze > requirements-freeze.txt
	@echo "$(GREEN)✅ Dependencias guardadas en requirements-freeze.txt$(NC)"

info: ## Muestra información del proyecto
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║          📊 INFORMACIÓN DEL PROYECTO                        ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Nombre:$(NC) Clasificador de Frutas CNN"
	@echo "$(GREEN)Tecnologías:$(NC) TensorFlow, Keras, Flask"
	@echo "$(GREEN)Clases:$(NC) manzana, banano, mango, naranja, pera"
	@echo ""
	@echo "$(BLUE)Archivos principales:$(NC)"
	@ls -lh scripts/*.py utils/*.py 2>/dev/null | awk '{printf "  %-30s %s\n", $$9, $$5}' || true
	@echo ""
	@if [ -d "dataset/processed" ]; then \
		echo "$(GREEN)✅ Dataset procesado existe$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  Dataset no procesado$(NC)"; \
	fi
	@if [ -f "models/fruit_classifier.h5" ]; then \
		echo "$(GREEN)✅ Modelo entrenado existe$(NC)"; \
		ls -lh models/fruit_classifier.h5 | awk '{printf "   Tamaño: %s\n", $$5}'; \
	else \
		echo "$(YELLOW)⚠️  Modelo no entrenado$(NC)"; \
	fi
	@echo ""

quick-start: install verify setup-dataset ## Inicio rápido completo
	@echo "$(GREEN)✅ Configuración inicial completada$(NC)"
	@echo "$(YELLOW)Próximos pasos:$(NC)"
	@echo "  1. Agrega imágenes a dataset/raw/<clase>/"
	@echo "  2. make clean-data"
	@echo "  3. make train"
	@echo "  4. make run"

# Alias comunes
i: install ## Alias para install
v: verify ## Alias para verify
t: train ## Alias para train
r: run ## Alias para run
c: clean ## Alias para clean
