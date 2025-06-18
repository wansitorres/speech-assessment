FROM runpod/base:0.4.0-cuda11.8

# Set working directory
WORKDIR /

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY handler.py .

# Set environment variable placeholder (will be overridden by RunPod)
ENV HUGGINGFACE_TOKEN=""

# Command to run the handler
CMD ["python", "-u", "handler.py"]