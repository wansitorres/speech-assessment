# ─────────────────────────────────────────────
# Base: Python 3.10 Slim
# ─────────────────────────────────────────────
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# ─────────────────────────────────────────────
# Install system dependencies
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    libsndfile1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# Copy and install Python dependencies
# ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Copy application code
# ─────────────────────────────────────────────
COPY . .

# ─────────────────────────────────────────────
# Preload models
# ─────────────────────────────────────────────

# Preload Hugging Face Wav2Vec2 model
RUN python -c "from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; \
               Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h'); \
               Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h')"

# Preload pyannote diarization pipeline
RUN python -c "from pyannote.audio import Pipeline; \
               Pipeline.from_pretrained('pyannote/speaker-diarization@3.1')"

# Download ONNX vocal separation models
RUN mkdir -p models && \
    wget -O models/UVR-MDX-NET-Inst_HQ_3.onnx https://huggingface.co/innnky/UVR_MDXNET_Inst_HQ_3/resolve/main/UVR-MDX-NET-Inst_HQ_3.onnx && \
    wget -O models/UVR_MDXNET_KARA_2.onnx https://huggingface.co/innnky/UVR_MDXNET_KARA_2/resolve/main/UVR_MDXNET_KARA_2.onnx

# ─────────────────────────────────────────────
# Default command
# ─────────────────────────────────────────────
CMD ["python3", "handler.py"]
