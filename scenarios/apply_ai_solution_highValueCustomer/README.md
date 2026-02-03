# AI Model Application Scenario - High Value Customer Prediction

## Ownership
Authors: Harsh Gohel and Prajwal Vaidya

## Course Information
This image was created as part of the course **'M. Grum: Advanced AI-based Application Systems'** by the **Junior Chair for Business Information Science, esp. AI-based Application Systems** at University of Potsdam.

## Description
This docker-compose configuration applies the trained AI Neural Network model to activation data for high-value customer prediction.

## Components
1. **knowledgeBase**: Provides trained AI and OLS models (busybox)
2. **activationBase**: Provides customer activation data (busybox)
3. **codeBase**: Provides additional data resources (busybox)
4. **executor_ai**: Executes the AI prediction algorithm (python:3.9-slim)

## Docker Images Used
- harshgohel923/knowledgebase_highvaluecustomer:latest
- harshgohel923/activationbase_highvaluecustomer:latest
- harshgohel923/codebase_highvaluecustomer:latest

## Prerequisites
- Docker and Docker Compose installed
- All required images pulled from Docker Hub
- External volume 'ai_system' created

## Usage

### 1. Pull Required Images
```bash
docker pull harshgohel923/knowledgebase_highvaluecustomer:latest
docker pull harshgohel923/activationbase_highvaluecustomer:latest
docker pull harshgohel923/codebase_highvaluecustomer:latest
```

### 2. Create External Volume
```bash
docker volume create ai_system
```

### 3. Clean Any Existing Volume Data
```bash
docker volume rm ai_system 2>/dev/null || true
docker volume create ai_system
```

### 4. Run Application
```bash
docker-compose up
```

### 5. View Logs
```bash
docker-compose logs -f executor_ai
```

### 6. Stop Application
```bash
docker-compose down
```

## Expected Output
Predictions will be saved to the shared volume at `/tmp/predictions/ai_predictions.csv`.

## License
AGPL-3.0