# Code Base - High Value Customer Prediction

## Ownership
Authors: Harsh Gohel and Prajwal Vaidya

## Course Information
This image was created as part of the course **'M. Grum: Advanced AI-based Application Systems'** by the **Junior Chair for Business Information Science, esp. AI-based Application Systems** at University of Potsdam.

## Data Origin
Original Dataset: High Value Customer from GitHub repository

## Description
This image contains the execution code for applying trained AI and OLS models to predict high-value customers.

### Contents
- `RunAiModel.py`: Applies the AI Neural Network model
- `RunOLSModel.py`: Applies the OLS Regression model
- Pre-installed dependencies: TensorFlow, pandas, numpy, scikit-learn, statsmodels

### Usage
This image is designed to be used in docker-compose scenarios where it:
1. Loads models from `/tmp/knowledgeBase/`
2. Loads activation data from `/tmp/activationBase/`
3. Makes predictions and saves results to `/tmp/predictions/`

## License
This project is licensed under the AGPL-3.0 License.