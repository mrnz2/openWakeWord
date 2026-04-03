FROM python:3.10-slim

# STEP 1: System dependencies
RUN apt-get update && apt-get install -y \
    git libespeak-ng-dev gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

# STEP 2: Base python dependencies
RUN pip install --no-cache-dir \
    "numpy<2" \
    "scipy>=1.3,<2" \
    "tqdm>=4,<5" \
    "requests>=2,<3" \
    "pyyaml>=6,<7" \
    "scikit-learn>=1,<2" \
    "protobuf>=3.20,<5"

# STEP 3: Core ML stack
RUN pip install --no-cache-dir \
    "tensorflow==2.15.0" \
    "tensorflow-probability==0.23.0" \
    "onnxruntime>=1.10.0,<2" \
    "onnx==1.14.0" \
    "onnx-tf==1.10.0" \
    "onnx2tf==1.26.3"

# STEP 4: Audio/training stack
RUN pip install --no-cache-dir \
    "torch==2.2.1" \
    "torchaudio==2.2.1" \
    "torchinfo>=1.8,<2" \
    "torchmetrics>=0.11.4,<1" \
    "speechbrain>=0.5.14,<1" \
    "mutagen>=1.46,<2" \
    "audiomentations>=0.30,<1" \
    "torch-audiomentations>=0.11,<1" \
    "acoustics>=0.2.6,<1" \
    "pronouncing>=0.2,<1" \
    "julius" \
    "webrtcvad" \
    "espeak-phonemizer"

# STEP 5: Main package
RUN pip install --no-cache-dir "openwakeword==0.6.0"
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# STEP 6: Source and compatibility patch
WORKDIR /app
RUN git clone --branch v0.6.0 --depth 1 https://github.com/dscripka/openWakeWord.git .
RUN git clone https://github.com/dscripka/piper-sample-generator.git ./piper-sample-generator
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt','/app/piper-sample-generator/models/en-us-libritts-high.pt')"
RUN python -c "import os; os.makedirs('/app/openwakeword/resources/models', exist_ok=True); import openwakeword.utils as u; u.download_models(target_directory='/app/openwakeword/resources/models')"
COPY scripts/patch_openwakeword_train.py /tmp/patch_openwakeword_train.py
RUN python /tmp/patch_openwakeword_train.py /app/openwakeword/train.py

ENTRYPOINT ["python", "-m", "openwakeword.train"]