# Use a CUDA/PyTorch base image compatible with the selected RunPod GPU.
# PyTorch 2.7.1 + CUDA 12.8 supports Blackwell GPUs (RTX 50xx, B200) as well as older cards.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /worker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY handler.py .
ENV MODEL_ID=openbmb/VoxCPM2
CMD ["python", "-u", "handler.py"]
