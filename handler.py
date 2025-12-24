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
import re
from datetime import datetime, timedelta
from pathlib import Path
import psutil

import runpod
import torch
from TTS.api import TTS
from google.cloud import storage
import soundfile as sf
from pydub import AudioSegment

# Tentar importar deepspeed (opcional)
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except Exception as e:
    print(f"WARNING: DeepSpeed não disponível: {e}")
    print("Continuando sem DeepSpeed...")
    DEEPSPEED_AVAILABLE = False
    deepspeed = None


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

# Configuração de chunking para textos longos
# Valores baseados na documentação oficial do XTTS:
# - split_sentence usa 250 chars por padrão
# - gpt_max_text_tokens = 400 tokens (~400 chars considerando idiomas latinos)
CHUNK_THRESHOLD = 400  # Caracteres mínimos para ativar chunking
DEFAULT_CHUNK_SIZE = 250  # Tamanho alvo de cada chunk (recomendado pela doc XTTS)
PROGRESS_UPDATE_INTERVAL = 10  # Atualizar progresso a cada 10%

print("Inicializando XTTS V2...")


# ===== INICIALIZAÇÃO DO MODELO =====

# Detectar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Carregar modelo XTTS V2
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("Modelo XTTS V2 carregado com sucesso!")
    
    # Ativar DeepSpeed inference para aceleração (se disponível)
    if device == "cuda" and tts is not None and DEEPSPEED_AVAILABLE:
        print("Inicializando DeepSpeed inference...")
        try:
            tts.synthesizer.tts_model = deepspeed.init_inference(
                tts.synthesizer.tts_model,
                mp_size=1,
                dtype=torch.float16,
                replace_with_kernel_inject=True
            )
            print("✓ DeepSpeed ativado com sucesso!")
        except Exception as ds_error:
            print(f"WARNING: Não foi possível ativar DeepSpeed: {ds_error}")
            print("Continuando com inferência padrão...")
    elif device == "cuda" and not DEEPSPEED_AVAILABLE:
        print("DeepSpeed não está disponível. Usando inferência padrão do PyTorch.")
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


# ===== FUNÇÕES DE CHUNKING E PROGRESS =====

def split_text_into_sentence_chunks(text, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Divide texto em chunks respeitando limites de sentenças.
    Cada chunk contém sentenças completas e tem aproximadamente chunk_size caracteres.
    
    Valor padrão de 250 chars é baseado na função split_sentence do XTTS,
    que usa text_split_length=250 como padrão.
    """
    # Dividir texto em sentenças usando regex
    # Captura pontos finais seguidos de espaço/quebra de linha
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Se adicionar esta sentença não exceder muito o limite, adiciona
        if len(current_chunk) + len(sentence) + 1 <= chunk_size * 1.3:  # 30% de tolerância
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # Se o chunk atual está vazio e a sentença é muito longa, adiciona sozinha
            if not current_chunk:
                chunks.append(sentence)
            else:
                # Salva chunk atual e começa novo com esta sentença
                chunks.append(current_chunk)
                current_chunk = sentence
    
    # Adicionar último chunk se houver
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"Texto dividido em {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} caracteres")
    
    return chunks


def save_progress_to_gcs(job_id, progress_data):
    """
    Salva progresso do job no GCS para permitir polling externo.
    """
    if not gcs_client or not GCS_BUCKET_NAME:
        return
    
    try:
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"progress/{job_id}.json")
        
        progress_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
        blob.upload_from_string(
            json.dumps(progress_data, indent=2),
            content_type="application/json"
        )
    except Exception as e:
        print(f"WARNING: Erro ao salvar progresso no GCS: {e}")


def delete_progress_from_gcs(job_id):
    """
    Remove arquivo de progresso do GCS após conclusão do job.
    """
    if not gcs_client or not GCS_BUCKET_NAME:
        return
    
    try:
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"progress/{job_id}.json")
        blob.delete()
        print(f"Progresso deletado do GCS: {job_id}")
    except Exception as e:
        print(f"WARNING: Erro ao deletar progresso do GCS: {e}")


def process_chunks_sequentially(chunks, job_id, ref_audio_path, language, 
                               temperature, speed, top_k, top_p, voice_id):
    """
    Processa chunks de texto sequencialmente, gerando áudio para cada um
    e reportando progresso.
    """
    chunk_files = []
    chunk_durations = []
    total_chunks = len(chunks)
    last_progress_percent = 0
    
    print(f"\nProcessando {total_chunks} chunks sequencialmente...")
    
    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        print(f"\n--- Chunk {chunk_num}/{total_chunks} ---")
        print(f"Texto: {chunk_text[:80]}...")
        
        # Gerar áudio para este chunk
        chunk_file = f"/tmp/chunk_{job_id}_{i}.wav"
        chunk_start = time.time()
        
        tts.tts_to_file(
            text=chunk_text,
            speaker_wav=ref_audio_path,
            language=language,
            file_path=chunk_file,
            temperature=temperature,
            speed=speed,
            top_k=top_k,
            top_p=top_p
        )
        
        chunk_time = time.time() - chunk_start
        
        # Obter duração do chunk
        audio_data, sample_rate = sf.read(chunk_file)
        chunk_duration = len(audio_data) / sample_rate
        
        chunk_files.append(chunk_file)
        chunk_durations.append(round(chunk_duration, 2))
        
        print(f"Chunk {chunk_num} gerado em {chunk_time:.2f}s (duração: {chunk_duration:.2f}s)")
        
        # Calcular progresso (90% para geração, 10% para concatenação)
        current_percent = int((chunk_num / total_chunks) * 90)
        
        # Atualizar progresso no RunPod e GCS a cada PROGRESS_UPDATE_INTERVAL%
        if current_percent - last_progress_percent >= PROGRESS_UPDATE_INTERVAL:
            progress_data = {
                "job_id": job_id,
                "status": "processing",
                "stage": "generating_chunks",
                "current_chunk": chunk_num,
                "total_chunks": total_chunks,
                "percent": current_percent,
                "chunks_completed": chunk_num,
                "chunk_durations": chunk_durations
            }
            
            # Atualizar RunPod progress
            try:
                runpod.serverless.progress_update(
                    job_id,
                    {
                        "stage": "generating_chunks",
                        "current_chunk": chunk_num,
                        "total_chunks": total_chunks,
                        "percent": current_percent
                    }
                )
            except Exception as e:
                print(f"WARNING: Erro ao atualizar progresso RunPod: {e}")
            
            # Salvar no GCS
            save_progress_to_gcs(job_id, progress_data)
            last_progress_percent = current_percent
    
    print(f"\n✓ Todos os {total_chunks} chunks gerados!")
    return chunk_files, chunk_durations


def concatenate_audio_chunks(chunk_files, job_id, voice_id):
    """
    Concatena múltiplos arquivos de áudio em um único arquivo,
    aplicando crossfade suave entre chunks.
    """
    print(f"\nConcatenando {len(chunk_files)} chunks de áudio...")
    
    # Atualizar progresso para concatenação
    try:
        runpod.serverless.progress_update(
            job_id,
            {
                "stage": "concatenating",
                "percent": 92
            }
        )
    except:
        pass
    
    concat_start = time.time()
    
    # Carregar primeiro chunk
    combined = AudioSegment.from_wav(chunk_files[0])
    
    # Adicionar demais chunks com crossfade
    for i, chunk_file in enumerate(chunk_files[1:], 1):
        print(f"  Concatenando chunk {i+1}/{len(chunk_files)}...")
        next_chunk = AudioSegment.from_wav(chunk_file)
        combined = combined.append(next_chunk, crossfade=30)  # 30ms de crossfade
    
    # Exportar resultado final
    output_path = f"/tmp/output_{voice_id}_{uuid.uuid4().hex[:8]}.wav"
    combined.export(
        output_path,
        format="wav",
        parameters=["-ar", "24000"]  # 24kHz sample rate
    )
    
    concat_time = time.time() - concat_start
    print(f"✓ Concatenação concluída em {concat_time:.2f}s")
    
    # Limpar arquivos temporários de chunks
    for chunk_file in chunk_files:
        try:
            os.remove(chunk_file)
        except:
            pass
    
    # Atualizar progresso final antes do upload
    try:
        runpod.serverless.progress_update(
            job_id,
            {
                "stage": "uploading",
                "percent": 95
            }
        )
    except:
        pass
    
    return output_path


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
        chunk_size = int(job_input.get("chunk_size", DEFAULT_CHUNK_SIZE))
        
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
        
        # Gerar job_id único para tracking
        job_id = job.get("id", str(uuid.uuid4()))
        text_length = len(gen_text)
        use_chunking = text_length > CHUNK_THRESHOLD
        
        print(f"\n=== Nova solicitação ===")
        print(f"Job ID: {job_id}")
        print(f"Texto: {text_length} caracteres - {gen_text[:100]}...")
        print(f"Voice ID: {voice_id}")
        print(f"Idioma: {language}")
        print(f"Estratégia: {'CHUNKED' if use_chunking else 'SINGLE'}")
        print(f"Parâmetros: temp={temperature}, speed={speed}, top_k={top_k}, top_p={top_p}")
        
        # Verificar se o modelo está carregado
        if tts is None:
            return {"error": "Modelo XTTS V2 não está carregado"}
        
        # Obter áudio de referência (cache ou download)
        ref_audio_path = get_cached_audio(voice_id)
        if not ref_audio_path:
            print("Áudio não está em cache. Baixando do GCS...")
            ref_audio_path = download_audio_from_gcs(ref_audio_url, voice_id)
        
        start_time = time.time()
        chunk_info = []
        
        # Escolher estratégia baseado no tamanho do texto
        if use_chunking:
            # Processar em chunks para textos longos
            print(f"\nTexto longo detectado ({text_length} chars). Usando estratégia CHUNKED.")
            
            # Dividir texto em chunks
            chunks = split_text_into_sentence_chunks(gen_text, chunk_size)
            
            # Processar chunks sequencialmente
            chunk_files, chunk_durations = process_chunks_sequentially(
                chunks, job_id, ref_audio_path, language,
                temperature, speed, top_k, top_p, voice_id
            )
            
            # Concatenar todos os chunks
            output_path = concatenate_audio_chunks(chunk_files, job_id, voice_id)
            
            # Construir info dos chunks
            for i, (chunk_text, duration) in enumerate(zip(chunks, chunk_durations)):
                chunk_info.append({
                    "chunk": i + 1,
                    "characters": len(chunk_text),
                    "duration": duration
                })
        else:
            # Processar de uma vez para textos curtos
            print(f"\nTexto curto ({text_length} chars). Usando estratégia SINGLE.")
            output_path = f"/tmp/output_{voice_id}_{uuid.uuid4().hex[:8]}.wav"
            
            print("Gerando áudio com XTTS V2...")
            
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
        print(f"\nÁudio gerado em {generation_time:.2f}s")
        
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
        
        # Deletar arquivo de progresso do GCS
        delete_progress_from_gcs(job_id)
        
        # Retornar resultado
        result = {
            "audio_url": audio_url,
            "duration": round(duration, 2),
            "voice_id": voice_id,
            "language": language,
            "generation_time": round(generation_time, 2),
            "processing_strategy": "chunked" if use_chunking else "single",
            "text_length": text_length,
            "parameters": {
                "temperature": temperature,
                "speed": speed,
                "top_k": top_k,
                "top_p": top_p
            }
        }
        
        # Adicionar info de chunks se aplicável
        if use_chunking and chunk_info:
            result["chunks_processed"] = len(chunk_info)
            result["chunk_info"] = chunk_info
        
        print(f"✓ Sucesso! Duração: {duration:.2f}s")
        return result
        
    except Exception as e:
        print(f"ERRO no handler: {e}")
        import traceback
        traceback.print_exc()
        
        # Limpar progresso do GCS em caso de erro
        try:
            job_id = job.get("id", "unknown")
            delete_progress_from_gcs(job_id)
        except:
            pass
        
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
