FROM busybox:latest

# Create directory structure
RUN mkdir -p /tmp/activationBase

# Copy activation data
COPY activation_data.csv /tmp/activationBase/

# Copy README
COPY README.md /tmp/activationBase/

# Set working directory
WORKDIR /tmp/activationBase