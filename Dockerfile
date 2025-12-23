# Dockerfile para XTTS V2 no RunPod
# Imagem base com PyTorch e CUDA
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    sox \
    libsox-fmt-all \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    transformers==4.33.0 \
    TTS>=0.22.0 \
    google-cloud-storage>=2.10.0 \
    torch>=2.0.0 \
    torchaudio \
    soundfile \
    psutil \
    numpy \
    pydub

# Aceitar termos de serviço do Coqui TTS
ENV COQUI_TOS_AGREED=1

# Pré-baixar modelos XTTS v2 para acelerar o startup
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)"

# Criar diretório para cache de áudios
RUN mkdir -p /tmp/audio_cache

# Copiar script de entrada
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Variável de ambiente para o repositório GitHub
ENV GITHUB_REPO_URL=""

# Expor porta (opcional, RunPod gerencia isso)
EXPOSE 8000

# Definir entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
