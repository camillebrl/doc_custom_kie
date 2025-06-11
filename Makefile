SHELL := /bin/bash

TIMESTAMP := $(shell date +%s)
NAME ?= results_$(TIMESTAMP)

.PHONY: help deps annotate_data finetuning_model final_inference all

help:
	@echo "Usage:"
	@echo "  make NAME=<project_name> deps              # installer les paquets système"
	@echo "  make NAME=<project_name> annotate_data     # lancer l'app Flask d'annotation"
	@echo "  make NAME=<project_name> finetuning_model  # fine-tuning, output_dir=NAME"
	@echo "  make NAME=<project_name> final_inference   # lancer l'inference, model_path=NAME/final_model"
	@echo "  make NAME=<project_name> all               # exécute deps → annotate_data → finetuning_model → final_inference"

deps:
	sudo apt-get update
	sudo apt-get install -y tesseract-ocr libtesseract-dev \
	                       tesseract-ocr-eng tesseract-ocr-fra
	poetry install

annotate_data:
	poetry run python -m doc_custom_extraction.annotate

finetuning_model:
	@echo "Fine-tuning LayoutLMv3 → modèle final et checkpoints stockés dans '$(NAME)'"
	poetry run python -m doc_custom_extraction.layoutlmv3_ft --output_dir $(NAME) --save_label_data

final_inference:
	@echo "Lancement de l'inférence avec le modèle '$(NAME)/final_model'"
	poetry run python -m doc_custom_extraction.inference --model_path $(NAME)/final_model

all_no_install: annotate_data finetuning_model final_inference

all: deps annotate_data finetuning_model final_inference
