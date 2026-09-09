# Root Dockerfile for RunPod GitHub integration.
# RunPod builds this file with the repository root as its context.
# PyTorch 2.7.1 + CUDA 12.8 supports Blackwell GPUs (RTX 50xx, B200) as well as older cards.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /worker
# ffmpeg gives librosa/audioread a backend so compressed reference audio
# (M4A/AAC/WebM/Opus) uploaded by the desktop client can be decoded to WAV.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY worker/handler.py ./handler.py
ENV MODEL_ID=openbmb/VoxCPM2
CMD ["python", "-u", "handler.py"]
