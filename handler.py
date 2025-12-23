"""
Handler RunPod para XTTS V2 Multilanguage
Suporta cache de áudios de referência, parâmetros avançados de síntese,
e integração completa com Google Cloud Storage.
"""

import os
import json
import time
import uuid
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import psutil

import runpod
import torch
from TTS.api import TTS
from google.cloud import storage
import soundfile as sf


# ===== CONFIGURAÇÃO =====

# Diretório de cache local para áudios de referência
CACHE_DIR = Path("/tmp/audio_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Limite de cache (5GB em bytes)
CACHE_SIZE_LIMIT = 5 * 1024 * 1024 * 1024

# Configuração GCS
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_CREDENTIALS = os.getenv("GCS_CREDENTIALS")

# Idiomas suportados pelo XTTS V2
SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", 
    "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"
]

# Valores padrão dos parâmetros
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SPEED = 1.0
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.85

print("Inicializando XTTS V2...")


# ===== INICIALIZAÇÃO DO MODELO =====

# Detectar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Carregar modelo XTTS V2
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("Modelo XTTS V2 carregado com sucesso!")
except Exception as e:
    print(f"ERRO ao carregar modelo: {e}")
    tts = None


# ===== CLIENTE GCS =====

def init_gcs_client():
    """Inicializa cliente do Google Cloud Storage"""
    if not GCS_CREDENTIALS:
        print("WARNING: GCS_CREDENTIALS não configurado")
        return None
    
    try:
        # Salvar credenciais em arquivo temporário
        creds_path = "/tmp/gcs_credentials.json"
        with open(creds_path, "w") as f:
            f.write(GCS_CREDENTIALS)
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        client = storage.Client()
        print("Cliente GCS inicializado com sucesso!")
        return client
    except Exception as e:
        print(f"ERRO ao inicializar GCS: {e}")
        return None


gcs_client = init_gcs_client()


# ===== FUNÇÕES DE CACHE =====

def get_cache_size():
    """Retorna o tamanho total do cache em bytes"""
    total_size = 0
    for file_path in CACHE_DIR.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


def clean_cache():
    """Remove arquivos mais antigos do cache até ficar abaixo do limite"""
    current_size = get_cache_size()
    
    if current_size <= CACHE_SIZE_LIMIT:
        return
    
    print(f"Cache excedeu limite ({current_size / (1024**3):.2f}GB). Limpando...")
    
    # Listar todos os arquivos com seu timestamp de modificação
    files = []
    for file_path in CACHE_DIR.rglob("*"):
        if file_path.is_file():
            files.append((file_path, file_path.stat().st_mtime))
    
    # Ordenar por data de modificação (mais antigos primeiro)
    files.sort(key=lambda x: x[1])
    
    # Remover arquivos até ficar abaixo do limite
    for file_path, _ in files:
        if current_size <= CACHE_SIZE_LIMIT:
            break
        
        file_size = file_path.stat().st_size
        try:
            file_path.unlink()
            current_size -= file_size
            print(f"Removido do cache: {file_path.name}")
        except Exception as e:
            print(f"Erro ao remover {file_path}: {e}")
    
    print(f"Cache limpo. Tamanho atual: {current_size / (1024**3):.2f}GB")


def get_cached_audio(voice_id):
    """Retorna o caminho do áudio em cache ou None se não existir"""
    cache_path = CACHE_DIR / f"{voice_id}.wav"
    if cache_path.exists():
        print(f"Áudio encontrado em cache: {voice_id}")
        return str(cache_path)
    return None


def download_audio_from_gcs(ref_audio_url, voice_id, max_retries=3):
    """
    Baixa áudio de referência do GCS com retry automático
    Retorna o caminho local do arquivo
    """
    cache_path = CACHE_DIR / f"{voice_id}.wav"
    
    # Verificar cache primeiro
    if cache_path.exists():
        return str(cache_path)
    
    if not gcs_client or not GCS_BUCKET_NAME:
        raise ValueError("GCS não configurado corretamente")
    
    # Extrair blob name da URL (assumindo formato gs://bucket/path ou URL assinada)
    if ref_audio_url.startswith("gs://"):
        # Formato: gs://bucket/path/to/file.wav
        blob_name = ref_audio_url.replace(f"gs://{GCS_BUCKET_NAME}/", "")
    elif "storage.googleapis.com" in ref_audio_url:
        # URL assinada ou pública
        # Extrair o path do blob
        parts = ref_audio_url.split(f"{GCS_BUCKET_NAME}/")
        if len(parts) > 1:
            blob_name = parts[1].split("?")[0]
        else:
            raise ValueError(f"URL GCS inválida: {ref_audio_url}")
    else:
        raise ValueError(f"Formato de URL não suportado: {ref_audio_url}")
    
    # Retry com backoff exponencial
    for attempt in range(max_retries):
        try:
            bucket = gcs_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(blob_name)
            
            print(f"Baixando áudio de referência (tentativa {attempt + 1}/{max_retries})...")
            blob.download_to_filename(str(cache_path))
            
            print(f"Áudio baixado e salvo em cache: {voice_id}")
            
            # Limpar cache se necessário
            clean_cache()
            
            return str(cache_path)
            
        except Exception as e:
            wait_time = 2 ** attempt  # Backoff exponencial: 1s, 2s, 4s
            print(f"Erro ao baixar áudio (tentativa {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                print(f"Aguardando {wait_time}s antes de tentar novamente...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Falha ao baixar áudio após {max_retries} tentativas")


def upload_audio_to_gcs(local_path, voice_id, max_retries=3):
    """
    Faz upload do áudio gerado para o GCS com retry automático
    Retorna a URL assinada com validade de 24h
    """
    if not gcs_client or not GCS_BUCKET_NAME:
        raise ValueError("GCS não configurado corretamente")
    
    # Gerar nome único para o arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    blob_name = f"generated/{voice_id}_{timestamp}_{unique_id}.wav"
    
    # Retry com backoff exponencial
    for attempt in range(max_retries):
        try:
            bucket = gcs_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(blob_name)
            
            print(f"Fazendo upload do áudio gerado (tentativa {attempt + 1}/{max_retries})...")
            blob.upload_from_filename(local_path)
            
            # Gerar URL assinada com validade de 24h
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=24),
                method="GET"
            )
            
            print(f"Upload concluído: {blob_name}")
            return signed_url
            
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"Erro ao fazer upload (tentativa {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                print(f"Aguardando {wait_time}s antes de tentar novamente...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Falha ao fazer upload após {max_retries} tentativas")


# ===== HANDLER PRINCIPAL =====

def handler(job):
    """
    Handler principal do RunPod para XTTS V2
    
    Input esperado:
    {
        "input": {
            "gen_text": "Texto para ser sintetizado",
            "ref_audio_url": "gs://bucket/path/reference.wav ou URL assinada",
            "voice_id": "identificador_unico_voz",
            "language": "pt" (opcional, padrão: "pt"),
            "temperature": 0.7 (opcional, 0.1-1.0),
            "speed": 1.0 (opcional, 0.5-2.0),
            "top_k": 50 (opcional),
            "top_p": 0.85 (opcional)
        }
    }
    
    Output:
    {
        "audio_url": "URL assinada do GCS",
        "duration": duracao_em_segundos,
        "voice_id": "identificador",
        "language": "pt"
    }
    """
    try:
        job_input = job["input"]
        
        # Validar campos obrigatórios
        required_fields = ["gen_text", "ref_audio_url", "voice_id"]
        for field in required_fields:
            if field not in job_input:
                return {"error": f"Campo obrigatório ausente: {field}"}
        
        gen_text = job_input["gen_text"]
        ref_audio_url = job_input["ref_audio_url"]
        voice_id = job_input["voice_id"]
        language = job_input.get("language", "pt").lower()
        
        # Parâmetros opcionais de síntese
        temperature = float(job_input.get("temperature", DEFAULT_TEMPERATURE))
        speed = float(job_input.get("speed", DEFAULT_SPEED))
        top_k = int(job_input.get("top_k", DEFAULT_TOP_K))
        top_p = float(job_input.get("top_p", DEFAULT_TOP_P))
        
        # Validar idioma
        if language not in SUPPORTED_LANGUAGES:
            return {
                "error": f"Idioma não suportado: {language}",
                "supported_languages": SUPPORTED_LANGUAGES
            }
        
        # Validar parâmetros
        if not (0.1 <= temperature <= 1.0):
            return {"error": "temperature deve estar entre 0.1 e 1.0"}
        
        if not (0.5 <= speed <= 2.0):
            return {"error": "speed deve estar entre 0.5 e 2.0"}
        
        print(f"\n=== Nova solicitação ===")
        print(f"Texto: {gen_text[:100]}...")
        print(f"Voice ID: {voice_id}")
        print(f"Idioma: {language}")
        print(f"Parâmetros: temp={temperature}, speed={speed}, top_k={top_k}, top_p={top_p}")
        
        # Verificar se o modelo está carregado
        if tts is None:
            return {"error": "Modelo XTTS V2 não está carregado"}
        
        # Obter áudio de referência (cache ou download)
        ref_audio_path = get_cached_audio(voice_id)
        if not ref_audio_path:
            print("Áudio não está em cache. Baixando do GCS...")
            ref_audio_path = download_audio_from_gcs(ref_audio_url, voice_id)
        
        # Gerar áudio com XTTS V2
        output_path = f"/tmp/output_{voice_id}_{uuid.uuid4().hex[:8]}.wav"
        
        print("Gerando áudio com XTTS V2...")
        start_time = time.time()
        
        tts.tts_to_file(
            text=gen_text,
            speaker_wav=ref_audio_path,
            language=language,
            file_path=output_path,
            temperature=temperature,
            speed=speed,
            top_k=top_k,
            top_p=top_p
        )
        
        generation_time = time.time() - start_time
        print(f"Áudio gerado em {generation_time:.2f}s")
        
        # Obter duração do áudio
        audio_data, sample_rate = sf.read(output_path)
        duration = len(audio_data) / sample_rate
        
        # Upload para GCS
        print("Fazendo upload para GCS...")
        audio_url = upload_audio_to_gcs(output_path, voice_id)
        
        # Limpar arquivo temporário
        try:
            os.remove(output_path)
        except:
            pass
        
        # Retornar resultado
        result = {
            "audio_url": audio_url,
            "duration": round(duration, 2),
            "voice_id": voice_id,
            "language": language,
            "generation_time": round(generation_time, 2),
            "parameters": {
                "temperature": temperature,
                "speed": speed,
                "top_k": top_k,
                "top_p": top_p
            }
        }
        
        print(f"✓ Sucesso! Duração: {duration:.2f}s")
        return result
        
    except Exception as e:
        print(f"ERRO no handler: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ===== INICIAR RUNPOD =====

if __name__ == "__main__":
    print("Iniciando RunPod serverless handler...")
    print(f"Dispositivo: {device}")
    print(f"GCS configurado: {gcs_client is not None}")
    print(f"Bucket: {GCS_BUCKET_NAME}")
    print(f"Idiomas suportados: {', '.join(SUPPORTED_LANGUAGES)}")
    print("\nAguardando jobs...\n")
    
    runpod.serverless.start({"handler": handler})
