# mlops_project_christa_toufic

This project implements an **Email Text Classification system** to identify phishing emails from safe emails. The project uses machine learning models trained on textual data, with hyperparameter tuning and experiment tracking managed through **MLflow**.


## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
  - [Clone the Repository](#clone-the-repository)
  - [Install Poetry](#install-poetry)
  - [Set Up the Environment](#set-up-the-environment)
  - [Run the Dataset and MLflow Setup Script](#run-the-dataset-and-mlflow-setup-script)
<!-- - [Running the Application](#running-the-application)
- [Model Evaluation and Tracking](#model-evaluation-and-tracking)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license) -->

---
## Introduction

This project classifies email texts into two categories:
- **Phishing Emails**
- **Safe Emails**

The system:
- Preprocesses and balances the dataset.
- Builds multiple text classification models using pipelines with **TF-IDF Vectorizer** and classifiers (SGD, Logistic Regression, Random Forest, etc.).
- Optimizes hyperparameters using Grid Search.
- Logs all experiments, metrics, and models to **MLflow** for tracking and evaluation.

---

## Features
- **Preprocessing Pipeline**: Merging datasets, cleaning data, sampling.
- **Two Classifiers**: SGD, Logistic Regression.
- **Experiment Tracking**: Integrated MLflow tracking for metrics, models, and artifacts.
- **Model Persistence**: Save the vectorizer and best model together.
- **Reproducibility**: Includes dataset and MLflow DB download script.

---

## Setup and Installation

Follow these steps to set up the project.

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/toufic-fy/mlops_project_christa_toufic.git
cd email-classifier
```

### 2. Install Poetry
This project uses Poetry for dependency management. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Verify the installation:
```bash
poetry --version
```
### 3. Set Up the Environment
**a. Install Project Dependencies**

Install the project's dependencies using Poetry:

```bash
poetry install
```
**b. Activate the Virtual Environment**

Activate the environment:

```bash
poetry shell
```
**c. Configure Environment Variables**

Create a .env file in the root directory:

```
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
DATASET_PATH=data/preprocessed_dataset.csv
```
### 4. Run the Dataset and MLflow Setup Script
You can run the below script to download both the dataset in csv format and the MLFlow experiment data from the initial notebook experiment:
```bash
poetry run data-download
```
You can also chose to download either the dataset only using the `--dataset-only` argument, or the MLFlow data only using the `--mlflow-data-only` argument. 
Use `--help` for detailed info.

### 5. Running Training Batch and Inference Batch

You can run the training pipeline on the dataset or the inference pipeline using the below commands:
```bash
poetry run email-classifier --script training
```
Use `--script inference` for inference.

---

### Model Evaluation and Tracking

### 1. Running Training Batch and Inference Batch

You can run the training pipeline on the dataset or the inference pipeline using the below commands:
```bash
poetry run email-classifier --script training
```
Use `--script inference` for inference.

### 2. Using Docker and Docker Compose

The project includes Docker and Docker Compose support for running the application and MLflow with minimal setup.

**a. Install Docker Install Docker and Docker Compose if they are not already installed:**

- **[Docker Installation Guide](https://docs.docker.com/engine/install/)**
- **[Docker Compose Installation Guide](https://docs.docker.com/compose/install/)**

**b. Persistent Data**

Ensure the data/ directory exists in your project root. This directory is used for:

- **Storing the SQLite database (mlflow.db).**
- **Storing MLflow artifacts (mlruns/).**

**c. Build and Run Containers**

To build and run the containers:
```bash
docker-compose up --build
```

This starts the following services:

- **Email Classifier API: Runs on http://localhost:8000.**
- **MLflow Tracking Server: Runs on http://localhost:5000.**

**d. Stop Services**

To stop the services, run:
```bash
docker-compose down
```