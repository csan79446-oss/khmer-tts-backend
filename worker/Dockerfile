# Use a CUDA/PyTorch base image compatible with the selected RunPod GPU.
# Verify the image's PyTorch and CUDA versions against the installed VoxCPM release.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /worker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY handler.py .
ENV MODEL_ID=openbmb/VoxCPM2
CMD ["python", "-u", "handler.py"]
