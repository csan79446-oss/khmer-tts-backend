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
COPY worker/handler.py worker/khmer_text_preparation.py worker/time_stretch.py ./
ENV MODEL_ID=openbmb/VoxCPM2
# Default TIME_STRETCH=1.3 ensures Khmer (and other out-of-domain scripts)
# are spoken at a natural pace. VoxCPM2 reads them ~1.2-1.5x too fast (1.3x is a balanced default) without
# this fix. Set TIME_STRETCH=0 to disable, or override per-request.
ENV TIME_STRETCH=1.3
CMD ["python", "-u", "handler.py"]
