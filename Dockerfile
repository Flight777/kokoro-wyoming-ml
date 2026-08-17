# Kokoro TTS — multilingual Wyoming server, CUDA (Blackwell-safe: onnxruntime, no torch)
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# kokoro-onnx[gpu] installs BOTH onnxruntime (CPU) and onnxruntime-gpu; when both
# are present the CPU build shadows the GPU one. Remove CPU, force GPU, then prove it.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y onnxruntime \
 && pip install --no-cache-dir --force-reinstall --no-deps "onnxruntime-gpu>=1.22.0" \
 && python3 -c "import onnxruntime as rt; p=rt.get_available_providers(); print('providers:', p); assert 'CUDAExecutionProvider' in p, 'CUDA provider missing'"

# Model + voices (v1.0 multilingual pack: 54 voices incl. zf_/zm_ Mandarin)
RUN wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
 && wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

COPY main.py .

ENV ONNX_PROVIDER=CUDAExecutionProvider \
    URI=tcp://0.0.0.0:10210 \
    KOKORO_VOICE=zf_xiaoxiao \
    KOKORO_SPEED=1.0

EXPOSE 10210
ENTRYPOINT ["python3", "main.py"]
