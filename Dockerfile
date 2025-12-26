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

# Copiar requirements.txt
COPY requirements.txt /app/requirements.txt

# Instalar dependências Python do requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Instalar DeepSpeed sem compilar ops CUDA (usa DS_BUILD_OPS=0)
# Isso permite usar DeepSpeed em modo básico sem precisar do CUDA toolkit
RUN DS_BUILD_OPS=0 pip install --no-cache-dir deepspeed>=0.12.0 || \
    echo "WARNING: DeepSpeed installation failed, continuing without it"

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
