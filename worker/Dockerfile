# Use a CUDA/PyTorch base image compatible with the selected RunPod GPU.
# PyTorch 2.7.1 + CUDA 12.8 supports Blackwell GPUs (RTX 50xx, B200) as well as older cards.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /worker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY handler.py khmer_text_preparation.py .
ENV MODEL_ID=openbmb/VoxCPM2
# Default TIME_STRETCH=1.5 ensures Khmer (and other out-of-domain scripts)
# are spoken at a natural pace. VoxCPM2 reads them ~1.2-1.5x too fast without
# this fix. Set TIME_STRETCH=0 to disable, or override per-request.
ENV TIME_STRETCH=1.5
CMD ["python", "-u", "handler.py"]
