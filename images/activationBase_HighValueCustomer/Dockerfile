FROM busybox

# Create directory
RUN mkdir -p /tmp/activationBase

# Copy data file
COPY activation_data.csv /tmp/activationBase/activation_data.csv

# Copy README
COPY README.md /tmp/activationBase/README.md

# Set working directory
WORKDIR /tmp/activationBase

# Keep container running
CMD ["sh"]
