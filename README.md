# Valuable Customer Prediction (AI-CPS Project)

**Course:** Advanced AI-Based Application Systems (AIBAS)
**University:** University of Potsdam
**Owner:** Prajwal Vaidya, Harsh Gohel

## 📌 Problem Statement

The objective of this project is to design, evaluate, and deploy a **reproducible AI-based system** that predicts whether an e-commerce customer belongs to a **high-value segment**.
High-value customers are identified based on behavioral, transactional, and engagement patterns, enabling applications such as targeted marketing, customer retention, and personalized offers.


## 🧠 Project Overview

This project implements and compares:

* **Artificial Neural Network (ANN)** — nonlinear, high-performance classifier
* **Ordinary Least Squares (OLS)** — linear baseline model

The system follows the **AI-CPS architecture**, separating data, models, and inference logic into independent Docker containers that communicate via a shared volume.

---

## 🏗️ Project Architecture

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

## 📂 Docker Images

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

## 🚀 How to Run (No GitHub Clone Required)

> **Only Docker is required.**
> No Python, no virtual environment, no repository cloning.

---

### 1️⃣ Pull Docker Images

```bash
docker pull harshgohel923/knowledgebase_highvaluecustomer
docker pull harshgohel923/activationbase_highvaluecustomer
docker pull harshgohel923/codebase_highvaluecustomer
```

---

### 2️⃣ Create Shared Volume

```bash
docker volume rm ai_system
docker volume create ai_system
```

---

### 3️⃣ Run Inference Manually

```bash
docker run --rm -v ai_system:/tmp harshgohel923/activationbase_highvaluecustomer
docker run --rm -v ai_system:/tmp harshgohel923/codebase_highvaluecustomer
```

---

### 4️⃣ View Prediction Output

```bash
docker run --rm -v ai_system:/tmp busybox cat /tmp/prediction_output.csv
```


---

## 🍎 Running on macOS (Default)

✔ Works **out-of-the-box** on:

* Apple Silicon (M1/M2/M3)

Docker automatically pulls the correct architecture when images are built as **multi-platform (`amd64`, `arm64`)**.

No configuration changes required.

---

## 🪟 Running on Windows (If Needed)

If platform mismatch occurs, add this under each service in `docker-compose.yml`:

```yaml
platform: linux/amd64
```

Then run:

```bash
docker-compose up
```

---

## 📊 Use Cases

* High-value customer identification
* Marketing campaign optimization
* Customer lifetime value analysis
* AI vs linear model comparison
* Reproducible AI deployment using Docker

---

## 📈 Models & Evaluation

* **ANN**

  * ROC AUC ≈ 0.99
  * Captures nonlinear customer behavior
* **OLS**

  * ROC AUC ≈ 0.89
  * Linear baseline for comparison

Training diagnostics and evaluation plots are stored in the `documentation/` directory.

---

## 👥 Contributions

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

## 📜 License

This project is released under the **AGPL-3.0 License**, in compliance with the requirements of the AIBAS course.

---

## ✅ Reproducibility Statement

This project can be fully reproduced by **pulling Docker images only**, without cloning the repository or installing additional dependencies.
