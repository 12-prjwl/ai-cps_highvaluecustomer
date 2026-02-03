# Use Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /codeBase

# Copy requirements and install dependencies
COPY requirements.txt /codeBase/
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python scripts
COPY RunAiModel.py /codeBase/
COPY RunOLSModel.py /codeBase/

# Copy README
COPY README.md /codeBase/

# Default command (will be overridden in docker-compose)
CMD ["python", "RunAiModel.py"]