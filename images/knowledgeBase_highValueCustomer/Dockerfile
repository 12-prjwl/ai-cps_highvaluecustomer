FROM busybox:latest

# Create directory structure
RUN mkdir -p /tmp/knowledgeBase

# Copy trained models
COPY currentAiSolution.h5 /tmp/knowledgeBase/
COPY currentOlsSolution.pkl /tmp/knowledgeBase/

# Copy README
COPY README.md /tmp/knowledgeBase/

# Set working directory
WORKDIR /tmp/knowledgeBase