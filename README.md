# Valuable Customer Prediction (AI-CPS Project)

## Ownership
Authors: Harsh Gohel and Prajwal Vaidya

## Course Information
This image was created as part of the course **'M. Grum: Advanced AI-based Application Systems'** by the **Junior Chair for Business Information Science, esp. AI-based Application Systems** at University of Potsdam.

## Problem Statement

The objective of this project is to design, evaluate, and deploy a **reproducible AI-based system** that predicts whether an e-commerce customer belongs to a **high-value segment**.
High-value customers are identified based on behavioral, transactional, and engagement patterns, enabling applications such as targeted marketing, customer retention, and personalized offers.


## Project Overview

This project implements and compares:

* **Artificial Neural Network (ANN)** — nonlinear, high-performance classifier
* **Ordinary Least Squares (OLS)** — linear baseline model

The system follows the **AI-CPS architecture**, separating data, models, and inference logic into independent Docker containers that communicate via a shared volume.

---

## Project Architecture

```
┌──────────────────┐
│ activationBase   │
│ activation_data  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ codeBase         │
│ inference logic  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ knowledgeBase    │
│ ANN + OLS models │
└──────────────────┘

Shared Docker Volume: ai_system  → mounted at /tmp
```

Each component is packaged as a **BusyBox-based Docker image** to ensure portability and minimal dependencies.

---

## Docker Images

Public Docker Hub images:

* **knowledgeBase**
  Contains trained ANN (`currentAiSolution.h5`) and OLS (`currentOlsSolution.pkl`) models.

* **activationBase**
  Contains `activation_data.csv` used for inference.

* **codeBase**
  Executes inference by loading models and activation data from `/tmp`.

All images include:

* Ownership information
* Course reference (AIBAS, University of Potsdam)
* Model/data description
* **AGPL-3.0 license commitment**

---

## How to Run

---

### 1️⃣ Install Docker Desktop (if not installed)
Download from: https://www.docker.com/products/docker-desktop

---

### 2️⃣ Clone this repository
```bash
cd ~/Desktop
git clone https://github.com/12-prjwl/AI-CPS_HighValueCustomer.git
cd AI-CPS_HighValueCustomer
```

---

### 3️⃣ Pull Docker Images

```bash
docker pull harshgohel923/knowledgebase_highvaluecustomer:latest
docker pull harshgohel923/activationbase_highvaluecustomer:latest
docker pull harshgohel923/codebase_highvaluecustomer:latest
```

---

### 4️⃣ Create Shared Volume

```bash
docker volume rm ai_system
docker volume create ai_system
```

---

### 5️⃣ Run AI prediction

```bash
cd scenarios/apply_ai_solution_highValueCustomer
docker-compose up
```

### After it completes, run OLS prediction
```bash
cd ../apply_ols_solution_highValueCustomer
docker-compose up
```

---

### 6️⃣ View Prediction Output

```bash
docker run --rm -v ai_system:/tmp busybox cat /tmp/prediction_output.csv
```


---

## Running on macOS (Apple silicon) (Default)

✔ Works **out-of-the-box** on:

* Apple Silicon (M1/M2/M3)

Docker automatically pulls the correct architecture when images are built.

No configuration changes required.

---

## Running on Windows (If Needed)

If platform mismatch occurs, add this under each service in `docker-compose.yml`:

```yaml
platform: linux/amd64
```

Then run:

```bash
docker-compose up
```

---

## Use Cases

* High-value customer identification
* Marketing campaign optimization
* Customer lifetime value analysis
* AI vs linear model comparison
* Reproducible AI deployment using Docker

---

## Models & Evaluation

* **ANN**

  * ROC AUC ≈ 0.99
  * Captures nonlinear customer behavior
* **OLS**

  * ROC AUC ≈ 0.89
  * Linear baseline for comparison

Training diagnostics and evaluation plots are stored in the `documentation/` directory.

---

## Contributions

**Prajwal Vaidya**

* Problem formulation and target engineering
* Data preprocessing and feature engineering
* ANN and OLS model development
* Evaluation, visualization, and activation testing

**Harsh Gohel**

* Docker image design and implementation
* Docker Compose orchestration
* BusyBox-based container setup
* Docker Hub publishing and documentation

---

## License

This project is released under the **AGPL-3.0 License**, in compliance with the requirements of the AIBAS course.

---

## Reproducibility Statement

This project can be fully reproduced by **pulling Docker images and using docker-compose.yml files located in scenarios folder**, without cloning the repository or installing additional dependencies.
