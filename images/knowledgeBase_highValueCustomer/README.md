# Knowledge Base - High Value Customer Prediction

## Ownership
Authors: Harsh Gohel and Prajwal Vaidya

## Course Information
This image was created as part of the course **'M. Grum: Advanced AI-based Application Systems'** by the **Junior Chair for Business Information Science, esp. AI-based Application Systems** at University of Potsdam.

## Description
This image contains trained AI and OLS models for predicting high-value customers based on customer behavior data.

### Model Characteristics

* **ANN**

  * ROC AUC ≈ 0.99
  * Captures nonlinear customer behavior
* **OLS**

  * ROC AUC ≈ 0.89
  * Linear baseline for comparison

Training diagnostics and evaluation plots are stored in the `documentation/` directory.


## Data Source
The models were trained on data scraped from GitHub repository

## License
This project is licensed under the AGPL-3.0 License.

## Usage
This image provides models at `/tmp/knowledgeBase/` for use in AI-based customer prediction systems.

Pull the image:
```bash
docker pull harshgohel923/knowledgebase_highvaluecustomer:latest
```