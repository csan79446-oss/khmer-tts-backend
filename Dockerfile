# Root Dockerfile for RunPod GitHub integration.
# RunPod builds this file with the repository root as its context.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /worker
COPY worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY worker/handler.py ./handler.py
ENV MODEL_ID=openbmb/VoxCPM2
CMD ["python", "-u", "handler.py"]
