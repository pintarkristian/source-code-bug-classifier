# Source Code Bug Classifier

> An end-to-end Machine Learning and AI project that classifies source-code snippets as **clean** or **potentially buggy/risky** using Pandas, TensorFlow, PyTorch, Hugging Face Transformers, FastAPI, Docker, and automated testing.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Transformer-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Baseline-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Inference-green)
![Status](https://img.shields.io/badge/Status-Portfolio_Project-brightgreen)

---

## 1. Project Overview

The goal of this project is to build a system that receives a source-code snippet and predicts whether the snippet is likely to be **clean** or **potentially buggy/risky**. The project compares two modeling approaches:

1. A **TensorFlow/Keras baseline model** trained on tokenized source code.
2. A **PyTorch Transformer model** fine-tuned using a CodeBERT-style architecture for code understanding.

The project also includes Pandas-based preprocessing, static code feature extraction, evaluation reports, a FastAPI inference service, Docker support, tests, and GitHub Actions CI.

---

## 2. Problem Statement

Modern software systems contain millions of lines of code, and even small bugs can lead to crashes, security vulnerabilities, data loss, or unexpected behavior. Traditional static analysis tools are useful, but they usually rely on predefined rules. Machine Learning and AI can add another layer by learning patterns from historical examples of buggy and clean code.

This project asks the following question:

> Can we train an AI system to classify source-code snippets as clean or potentially buggy/risky using both classic ML-style preprocessing and modern Transformer-based code models?

The project focuses on **risk classification**, not formal verification. The classifier should be treated as an assistant that highlights suspicious code, not as a replacement for human code review, secure coding practices, unit tests, or professional security scanners.

---

## 3. Why This Project Matters

This project is useful for a CV because it combines several skills that employers often look for in Machine Learning, AI, Data Science, and MLOps roles:

- Data preprocessing with **Pandas**
- Baseline model development with **TensorFlow/Keras**
- Transformer fine-tuning with **PyTorch** and **Hugging Face Transformers**
- Model evaluation with proper classification metrics
- API deployment with **FastAPI**
- Reproducible execution with **Docker**
- Test automation with **pytest**
- Continuous integration with **GitHub Actions**
- Clear documentation and project structure

Most beginner ML projects stop at a notebook. This project is intentionally structured as a production-style repository with reusable modules, command-line interfaces, evaluation scripts, tests, and a Dockerized API.

---

## 4. Architecture

```text
Input code snippet
        |
        v
Pandas preprocessing and cleaning
        |
        +-------------------------------+
        |                               |
        v                               v
Static feature extraction         Code tokenization
        |                               |
        v                               v
TensorFlow baseline model         PyTorch CodeBERT model
        |                               |
        +---------------+---------------+
                        |
                        v
                Evaluation reports
                        |
                        v
              FastAPI prediction API
                        |
                        v
                  Docker container
```

The system has two main modeling tracks:

### TensorFlow baseline track

The TensorFlow model uses source-code text as input and trains a lightweight deep-learning classifier. This baseline is useful because it gives a comparison point before using a larger Transformer model.

### PyTorch Transformer track

The PyTorch model uses a pretrained Transformer model suitable for code representation. The model is fine-tuned for binary classification: clean versus potentially buggy/risky.

### API inference track

After training, the best model can be loaded by a FastAPI application. The API receives a code snippet and returns a risk score, predicted label, model name, and optional heuristic notes.

---

## 5. Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.11+ |
| Data processing | Pandas, NumPy |
| Machine Learning | scikit-learn |
| Deep Learning baseline | TensorFlow, Keras |
| Transformer model | PyTorch, Hugging Face Transformers |
| Dataset loading | Hugging Face Datasets |
| API | FastAPI, Uvicorn, Pydantic |
| Evaluation | scikit-learn metrics, Matplotlib |
| Testing | pytest, httpx |
| Code quality | black, ruff |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## 6. Dataset

The recommended dataset for this project is a defect-detection dataset such as **CodeXGLUE Defect Detection**. The dataset should contain code snippets and binary labels.

Expected processed format:

```csv
code,label
"int add(int a, int b) { return a + b; }",0
"char buf[10]; strcpy(buf, input);",1
```

Label meaning:

```text
0 = clean / non-buggy
1 = potentially buggy / risky
```

The preprocessing pipeline normalizes the dataset into this simple format regardless of whether the raw data comes from a local CSV, JSONL file, or Hugging Face dataset.

---

## 7. Project Structure

```text
ai-powered-code-bug-classifier/
│
├── app/
│   ├── main.py
│   └── schemas.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── docker/
│   └── entrypoint.sh
│
├── models/
│   └── README.md
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── reports/
│   ├── metrics.json
│   └── figures/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── features.py
│   ├── train_tensorflow_baseline.py
│   ├── train_pytorch_transformer.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 8. Milestones

This repository is organized around 12 implementation milestones. Each milestone adds a clear, reviewable piece of functionality.

---

## Milestone 1 — Project Skeleton

The first milestone creates the complete project structure. This includes folders for source code, API code, tests, reports, models, notebooks, Docker files, and documentation.

### Goal

Create a clean repository layout that supports training, evaluation, inference, testing, and deployment.

### Main files

```text
app/main.py
app/schemas.py
src/config.py
src/data_preprocessing.py
src/features.py
src/train_tensorflow_baseline.py
src/train_pytorch_transformer.py
src/evaluate.py
src/predict.py
src/utils.py
tests/test_preprocessing.py
tests/test_features.py
tests/test_api.py
Dockerfile
docker-compose.yml
Makefile
requirements.txt
README.md
.gitignore
```

### Validation command

```bash
python -m compileall src app tests
```

---

## Milestone 2 — Dependencies and Developer Tooling

This milestone adds the Python dependencies and developer tooling required to build the project.

### Goal

Make the environment reproducible and easy to install.

### Core dependencies

```text
pandas
numpy
scikit-learn
tensorflow
torch
transformers
datasets
accelerate
evaluate
fastapi
uvicorn
pydantic
python-multipart
matplotlib
pytest
httpx
black
ruff
jupyter
```

### Local setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Validation command

```bash
make test
```

---

## Milestone 3 — Pandas Data Preprocessing

This milestone implements the data pipeline. The pipeline loads raw code data, cleans it, normalizes it, and saves train/validation/test splits.

### Goal

Convert raw data into clean CSV files that can be used by both TensorFlow and PyTorch models.

### Features

- Load CSV files
- Load JSONL files
- Load Hugging Face datasets
- Normalize columns to `code` and `label`
- Drop missing values
- Remove duplicate snippets
- Remove very short snippets
- Truncate very long snippets
- Convert labels to integers
- Create stratified train/validation/test splits
- Save processed CSV files

### Example command with local data

```bash
python -m src.data_preprocessing \
  --input data/raw/dataset.csv \
  --output-dir data/processed \
  --code-column code \
  --label-column label
```

### Example command with Hugging Face dataset

```bash
python -m src.data_preprocessing \
  --hf-dataset google/code_x_glue_cc_defect_detection \
  --output-dir data/processed
```

### Expected outputs

```text
data/processed/train.csv
data/processed/valid.csv
data/processed/test.csv
```

### Validation command

```bash
pytest tests/test_preprocessing.py -v
```

---

## Milestone 4 — Static Code Feature Extraction

This milestone adds classic ML-style feature extraction. Even though the final model uses deep learning, handcrafted features are useful for analysis, explainability, and baseline experiments.

### Goal

Extract numerical features from code snippets using Python and Pandas.

### Features extracted

```text
char_count
line_count
avg_line_length
function_call_count
loop_count
conditional_count
try_except_count
comment_line_count
suspicious_keyword_count
```

### Suspicious keywords

```text
eval
exec
pickle
subprocess
os.system
strcpy
sprintf
gets
input
shell=True
```

### Example usage

```python
from src.features import extract_features_from_code

code = "def divide(a, b): return a / b"
features = extract_features_from_code(code)
print(features)
```

### Validation command

```bash
pytest tests/test_features.py -v
```

---

## Milestone 5 — TensorFlow Baseline Model

This milestone builds the first trainable model. The TensorFlow baseline provides a comparison point for the Transformer model.

### Goal

Train a lightweight binary text classifier using TensorFlow/Keras.

### Model architecture

```text
Input code text
        |
        v
TextVectorization
        |
        v
Embedding
        |
        v
Bidirectional LSTM or Conv1D
        |
        v
Dense layers
        |
        v
Sigmoid output
```

### Training command

```bash
python -m src.train_tensorflow_baseline \
  --train data/processed/train.csv \
  --valid data/processed/valid.csv \
  --output models/tensorflow_baseline.keras \
  --epochs 3 \
  --batch-size 16
```

### Expected outputs

```text
models/tensorflow_baseline.keras
reports/tensorflow_metrics.json
```

### Metrics tracked

- Accuracy
- Precision
- Recall
- AUC
- Loss

---

## Milestone 6 — PyTorch Transformer Model

This milestone adds the AI-focused part of the project: a Transformer model fine-tuned for code classification.

### Goal

Fine-tune a pretrained code model for binary defect classification.

### Model approach

```text
Input code snippet
        |
        v
Transformer tokenizer
        |
        v
CodeBERT-style encoder
        |
        v
Classification head
        |
        v
Clean / Buggy prediction
```

### Training command

```bash
python -m src.train_pytorch_transformer \
  --train data/processed/train.csv \
  --valid data/processed/valid.csv \
  --model-name microsoft/codebert-base \
  --output-dir models/codebert-bug-classifier \
  --epochs 1 \
  --batch-size 8 \
  --max-length 256
```

### Expected outputs

```text
models/codebert-bug-classifier/
reports/codebert_metrics.json
```

### Metrics tracked

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The Transformer model is expected to perform better than the TensorFlow baseline when enough training data and compute are available.

---

## Milestone 7 — Evaluation Script

This milestone evaluates trained models on the test split.

### Goal

Measure model performance using a held-out test set and generate reproducible reports.

### Supported model types

```text
tensorflow
transformer
```

### Evaluation command for Transformer

```bash
python -m src.evaluate \
  --test data/processed/test.csv \
  --model-dir models/codebert-bug-classifier \
  --model-type transformer
```

### Evaluation command for TensorFlow

```bash
python -m src.evaluate \
  --test data/processed/test.csv \
  --model-dir models/tensorflow_baseline.keras \
  --model-type tensorflow
```

### Expected outputs

```text
reports/test_metrics.json
reports/figures/confusion_matrix.png
```

### Metrics

```text
accuracy
precision
recall
F1
ROC-AUC
PR-AUC
confusion matrix
```

---

## Milestone 8 — Prediction Logic

This milestone creates reusable prediction logic that can be used by the API, scripts, or future UI tools.

### Goal

Create a `CodeBugPredictor` class that loads a trained Transformer model and returns predictions.

### Example usage

```python
from src.predict import CodeBugPredictor

predictor = CodeBugPredictor("models/codebert-bug-classifier")
result = predictor.predict("def divide(a, b): return a / b")
print(result)
```

### Example response

```json
{
  "label": "buggy",
  "risk_score": 0.82,
  "model_name": "codebert",
  "notes": [
    "Potential division-by-zero risk",
    "No obvious denominator validation detected"
  ]
}
```

### Heuristic notes

The model prediction is primary. Heuristic notes are only explanatory. They may flag obvious risky patterns such as:

- `eval`
- `exec`
- `shell=True`
- `os.system`
- `pickle`
- `strcpy`
- `sprintf`
- `gets`
- division without an obvious zero check

---

## Milestone 9 — FastAPI Inference Service

This milestone exposes the classifier as an HTTP API.

### Goal

Allow users or applications to send code snippets to the model and receive predictions.

### Endpoints

```text
GET /health
POST /predict
```

### Run API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Health check

```bash
curl http://localhost:8000/health
```

### Prediction request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"code": "def divide(a, b): return a / b"}'
```

### Example response

```json
{
  "label": "buggy",
  "risk_score": 0.82,
  "model_name": "codebert",
  "notes": [
    "Potential division-by-zero risk"
  ]
}
```

### API tests

```bash
pytest tests/test_api.py -v
```

The API tests should use mocking so they do not require the full Transformer model to be downloaded during CI.

---

## Milestone 10 — Docker Support

This milestone adds CPU-friendly containerized execution using Docker and Docker Compose. The Docker image uses a Python slim base image, installs `requirements.txt`, sets `PYTHONUNBUFFERED=1`, sets `PYTHONPATH=/app`, copies the project into `/app`, exposes port `8000`, and runs the FastAPI application with Uvicorn by default.

### Goal

Make the project reproducible and easy to run on another machine without requiring GPU support.

### Build Docker image

```powershell
docker compose build
```

### Run API

```powershell
docker compose up api
```

### Run tests inside Docker

```powershell
docker compose run --rm api make test
```

### Run preprocessing inside Docker

```powershell
docker compose run --rm api make preprocess
```

### Train TensorFlow model inside Docker

```powershell
docker compose run --rm api make train-tf
```

### Train Transformer model inside Docker

```powershell
docker compose run --rm api make train-transformer
```

The `api` service mounts the local project directory into `/app` for development. The setup is CPU-friendly by default and does not require NVIDIA Docker, CUDA, or GPU configuration.

---

## Milestone 11 — README and Documentation Polish

This milestone turns the repository into a professional CV project.

### Goal

Make the GitHub repository easy to understand for recruiters, hiring managers, and technical interviewers.

### README sections

- Project overview
- Problem statement
- Architecture
- Tech stack
- Dataset
- Setup instructions
- Docker instructions
- Training commands
- Evaluation commands
- API usage
- Metrics table
- Project structure
- Limitations
- Future improvements
- CV bullet points

A strong README should answer three questions quickly:

1. What does this project do?
2. How do I run it?
3. Why is it technically impressive?

---

## Milestone 12 — GitHub Actions CI

This milestone adds automated checks to the repository.

### Goal

Run tests and code-quality checks automatically on every push and pull request.

### Workflow file

```text
.github/workflows/ci.yml
```

### CI checks

```bash
ruff check src app tests
black --check src app tests
pytest tests -v
```

### Important CI rule

Do not train models in CI. Do not download large datasets in CI. CI should stay fast and reliable.

---

# 9. Makefile Commands

The project should provide simple commands through a `Makefile`.

```makefile
install:
	pip install -r requirements.txt

preprocess:
	python -m src.data_preprocessing --hf-dataset google/code_x_glue_cc_defect_detection --output-dir data/processed

train-tf:
	python -m src.train_tensorflow_baseline --train data/processed/train.csv --valid data/processed/valid.csv --output models/tensorflow_baseline.keras --epochs 3 --batch-size 16

train-transformer:
	python -m src.train_pytorch_transformer --train data/processed/train.csv --valid data/processed/valid.csv --model-name microsoft/codebert-base --output-dir models/codebert-bug-classifier --epochs 1 --batch-size 8 --max-length 256

evaluate:
	python -m src.evaluate --test data/processed/test.csv --model-dir models/codebert-bug-classifier --model-type transformer

api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	pytest tests -v

format:
	black src app tests

lint:
	ruff check src app tests

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
```

---

# 10. Full Local Workflow

After implementing the milestones, the normal local workflow should look like this:

```bash
# 1. Create and activate environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
make install

# 3. Run tests
make test

# 4. Preprocess data
make preprocess

# 5. Train TensorFlow baseline
make train-tf

# 6. Train Transformer model
make train-transformer

# 7. Evaluate best model
make evaluate

# 8. Run API
make api
```

On Windows PowerShell, use:

```powershell
.venv\Scripts\Activate.ps1
```

instead of:

```bash
source .venv/bin/activate
```

---

# 11. Docker Workflow

Docker makes the project easier to reproduce. The `api` service is CPU-friendly, mounts the local project directory into `/app` for development, exposes port `8000`, and starts FastAPI with Uvicorn.

## Build the image

```powershell
docker compose build
```

## Run the API

```powershell
docker compose up api
```

## Run tests inside Docker

```powershell
docker compose run --rm api make test
```

## Run preprocessing inside Docker

```powershell
docker compose run --rm api make preprocess
```

## Train the TensorFlow baseline inside Docker

```powershell
docker compose run --rm api make train-tf
```

## Train the Transformer model inside Docker

```powershell
docker compose run --rm api make train-transformer
```

Once the API is running, open:

```text
http://localhost:8000/docs
```

FastAPI will automatically provide interactive API documentation.

---

# 12. API Documentation

## GET `/health`

Checks whether the service is running.

### Example request

```bash
curl http://localhost:8000/health
```

### Example response

```json
{
  "status": "ok",
  "model_loaded": true
}
```

## POST `/predict`

Classifies a code snippet.

### Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"code": "def divide(a, b): return a / b"}'
```

### Example response

```json
{
  "label": "buggy",
  "risk_score": 0.82,
  "model_name": "codebert",
  "notes": [
    "Potential division-by-zero risk",
    "No obvious denominator validation detected"
  ]
}
```

---

# 13. Evaluation Metrics

After training and evaluation, update this table with real results.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| TensorFlow baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| PyTorch CodeBERT | TBD | TBD | TBD | TBD | TBD | TBD |

## Metric interpretation

- **Accuracy**: Overall proportion of correct predictions.
- **Precision**: Of snippets predicted buggy, how many were actually buggy.
- **Recall**: Of actual buggy snippets, how many were detected.
- **F1-score**: Balance between precision and recall.
- **ROC-AUC**: Ranking quality across thresholds.
- **PR-AUC**: Useful when buggy samples are rare.

For bug detection, recall is often important because missing a risky snippet may be worse than flagging a clean one for review. However, the right threshold depends on the use case.

---

# 14. Example Prediction Scenarios

## Example 1 — Division risk

```python
def divide(a, b):
    return a / b
```

Possible output:

```json
{
  "label": "buggy",
  "risk_score": 0.74,
  "notes": ["Potential division-by-zero risk"]
}
```

## Example 2 — Safer division

```python
def divide(a, b):
    if b == 0:
        return None
    return a / b
```

Possible output:

```json
{
  "label": "clean",
  "risk_score": 0.21,
  "notes": []
}
```

## Example 3 — Unsafe evaluation

```python
def run_user_code(user_input):
    return eval(user_input)
```

Possible output:

```json
{
  "label": "buggy",
  "risk_score": 0.91,
  "notes": ["Use of eval detected"]
}
```

---

# 15. Testing Strategy

This project uses `pytest` for automated tests.

## Test categories

```text
tests/test_preprocessing.py
tests/test_features.py
tests/test_api.py
```

## Run all tests

```bash
pytest tests -v
```

## What should be tested

### Preprocessing tests

- Missing values are removed
- Duplicate snippets are removed
- Very short snippets are removed
- Labels are converted correctly
- Train/validation/test files are saved

### Feature extraction tests

- Empty code does not crash the extractor
- Line counts are correct
- Suspicious keywords are detected
- Function calls, conditionals, and loops are counted

### API tests

- `/health` returns status OK
- `/predict` accepts valid code
- `/predict` rejects empty code
- API tests use a mock predictor to avoid loading large models

---

# 16. Code Quality

This repository uses `black` for formatting and `ruff` for linting.

Format code:

```bash
make format
```

Lint code:

```bash
make lint
```

Run tests:

```bash
make test
```

A clean project should pass all three before committing code.

---

# 17. GitHub Actions CI

The repository should include a CI workflow at:

```text
.github/workflows/ci.yml
```

The workflow should run on:

```text
push
pull_request
```

The CI pipeline should:

```bash
ruff check src app tests
black --check src app tests
pytest tests -v
```

CI should not train models or download large datasets. Keep CI focused on fast validation.

---

# 18. Limitations

This project is an educational and portfolio-focused AI system. It has important limitations:

1. It does not prove that code is safe.
2. It may produce false positives and false negatives.
3. It depends heavily on the quality and distribution of the training dataset.
4. It may not generalize equally across all programming languages.
5. It should not replace secure code review, static analysis, unit testing, or penetration testing.
6. Transformer training may require significant compute for strong results.
7. Short snippets may lack enough context for accurate classification.

The correct interpretation is:

> This model estimates whether a code snippet appears risky based on learned patterns and simple heuristics.

---

# 19. Future Improvements

Possible extensions:

- Add support for multiple programming languages
- Add a Streamlit web interface
- Add SHAP or other explanation tools for baseline models
- Add more datasets for vulnerability classification
- Add GitHub pull request scanning
- Add batch file scanning
- Add model monitoring and drift detection
- Add experiment tracking with MLflow or Weights & Biases
- Add GPU-enabled Docker configuration
- Add ONNX export for faster inference
- Add threshold tuning for precision/recall control
- Add static analyzer comparison against tools such as Bandit or Semgrep

---

# 20. CV Bullet Points

Use one of these on your CV after completing the project.

## General Machine Learning version

> Built an end-to-end AI-powered code bug classifier using Python, Pandas, TensorFlow, PyTorch, Hugging Face Transformers, FastAPI, and Docker to classify source-code snippets as clean or potentially risky.

## AI Engineer version

> Fine-tuned a Transformer-based code model for defect classification and deployed it as a Dockerized FastAPI service with automated tests and reproducible training scripts.

## Machine Learning Engineer version

> Developed a full ML pipeline for code defect detection, including Pandas preprocessing, TensorFlow baseline modeling, PyTorch Transformer fine-tuning, evaluation reporting, and REST API inference.

## MLOps version

> Dockerized an end-to-end ML/AI system with training, evaluation, inference, testing, and CI workflows for reproducible code-risk classification.

## Strong version after adding real metrics

> Fine-tuned a CodeBERT-based Transformer for code defect classification, achieving X% F1-score on a held-out test set and deploying the model through a Dockerized FastAPI inference service.

---

# 21. Suggested Commit Plan

Use one commit per milestone.

```bash
git add .
git commit -m "Milestone 1: create project skeleton"

git add .
git commit -m "Milestone 2: add dependencies and tooling"

git add .
git commit -m "Milestone 3: implement data preprocessing"

git add .
git commit -m "Milestone 4: add static feature extraction"

git add .
git commit -m "Milestone 5: train TensorFlow baseline"

git add .
git commit -m "Milestone 6: fine-tune PyTorch transformer"

git add .
git commit -m "Milestone 7: add model evaluation"

git add .
git commit -m "Milestone 8: add prediction logic"

git add .
git commit -m "Milestone 9: add FastAPI inference service"

git add .
git commit -m "Milestone 10: add Docker support"

git add .
git commit -m "Milestone 11: polish README documentation"

git add .
git commit -m "Milestone 12: add GitHub Actions CI"
```

---

# 22. Final Project Checklist

Before publishing this repository on GitHub, verify the following:

- [ ] README is complete
- [ ] Project structure is clean
- [ ] Dependencies install correctly
- [ ] Preprocessing works
- [ ] TensorFlow baseline trains
- [ ] PyTorch Transformer trains
- [ ] Evaluation script generates metrics
- [ ] FastAPI app starts successfully
- [ ] Docker image builds successfully
- [ ] Tests pass locally
- [ ] GitHub Actions CI passes
- [ ] Large model files are not accidentally committed
- [ ] Dataset licensing is respected
- [ ] CV bullet points include real metrics after evaluation

---

# 23. Disclaimer

This project is for educational and portfolio purposes. It is not a formal security scanner and should not be used as the only method for detecting vulnerabilities or bugs in production systems. Always combine AI-based tools with human code review, tests, static analysis, and secure development practices.

---

# 24. License

Choose a license before publishing. For portfolio projects, the MIT License is a common choice.

```text
MIT License
```

---
