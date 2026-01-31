FROM busybox

# Create directories
RUN mkdir -p /tmp/learningBase/train
RUN mkdir -p /tmp/learningBase/validation

# Copy data files to appropriate locations
COPY training_data.csv /tmp/learningBase/train/training_data.csv
COPY test_data.csv /tmp/learningBase/validation/test_data.csv

# Copy README
COPY README.md /tmp/learningBase/README.md

# Set working directory
WORKDIR /tmp/learningBase

# Keep container running
CMD ["sh"]
