# XTTS V2 RunPod Serverless

Serviço serverless no RunPod para síntese de voz multilanguage usando XTTS V2 (Coqui TTS) com cache inteligente de áudios de referência e integração completa com Google Cloud Storage.

## 🚀 Características

- **Multilanguage**: Suporte para 16 idiomas (pt, en, es, fr, de, it, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko)
- **Voice Cloning**: Clonagem de voz usando áudio de referência
- **Cache Inteligente**: Cache local de áudios de referência com limpeza automática em 5GB
- **Parâmetros Avançados**: Controle fino de temperature, speed, top_k e top_p
- **Código Dinâmico**: Atualização automática via GitHub sem rebuild da imagem
- **GCS Integration**: Upload/download automático do Google Cloud Storage
- **GPU Accelerated**: Otimizado para GPUs CUDA

## 📋 Requisitos

- Conta no RunPod
- Google Cloud Platform com Storage habilitado
- Service Account com permissões de leitura/escrita no GCS
- Repositório GitHub (opcional, para código dinâmico)

## 🏗️ Estrutura do Projeto

```
xtts/
├── Dockerfile          # Imagem Docker com PyTorch + CUDA
├── entrypoint.sh       # Script de inicialização com git pull
├── handler.py          # Handler RunPod principal
├── requirements.txt    # Dependências Python
└── README.md          # Este arquivo
```

## 🔧 Configuração

### 1. Variáveis de Ambiente

Configure as seguintes variáveis no RunPod:

| Variável | Descrição | Obrigatório |
|----------|-----------|-------------|
| `GITHUB_REPO_URL` | URL do repositório GitHub para código dinâmico | Opcional |
| `GCS_CREDENTIALS` | JSON do Service Account do GCP | Sim |
| `GCS_BUCKET_NAME` | Nome do bucket do Google Cloud Storage | Sim |

**Exemplo de GCS_CREDENTIALS:**
```json
{
  "type": "service_account",
  "project_id": "seu-projeto",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "...",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "..."
}
```

### 2. Build da Imagem Docker

```bash
# Build
docker build -t seu-usuario/xtts-v2-runpod:latest .

# Push para Docker Hub
docker push seu-usuario/xtts-v2-runpod:latest
```

### 3. Deploy no RunPod

1. Acesse o RunPod Dashboard
2. Vá em "Serverless" → "New Endpoint"
3. Configure:
   - **Docker Image**: `seu-usuario/xtts-v2-runpod:latest`
   - **GPU Type**: RTX 4090, A100, ou similar
   - **Container Disk**: 10-20 GB
   - **Environment Variables**: Adicione as variáveis acima
4. Deploy!

## 📡 Uso da API

### Formato do Payload

```json
{
  "input": {
    "gen_text": "Olá! Este é um exemplo de síntese de voz usando XTTS V2.",
    "ref_audio_url": "gs://seu-bucket/referencias/voz_joao.wav",
    "voice_id": "joao_formal",
    "language": "pt",
    "temperature": 0.7,
    "speed": 1.0,
    "top_k": 50,
    "top_p": 0.85
  }
}
```

### Parâmetros

#### Obrigatórios

- **gen_text** (string): Texto para ser sintetizado
- **ref_audio_url** (string): URL do áudio de referência no GCS
  - Formato: `gs://bucket/path/file.wav` ou URL assinada
- **voice_id** (string): Identificador único da voz (usado para cache)

#### Opcionais

- **language** (string): Código do idioma (padrão: `pt`)
  - Suportados: `en`, `es`, `fr`, `de`, `it`, `pt`, `pl`, `tr`, `ru`, `nl`, `cs`, `ar`, `zh-cn`, `ja`, `hu`, `ko`
- **temperature** (float): Controle de variabilidade (0.1-1.0, padrão: 0.7)
  - Menor = mais consistente
  - Maior = mais expressivo
- **speed** (float): Velocidade da fala (0.5-2.0, padrão: 1.0)
- **top_k** (int): Top-K sampling (padrão: 50)
- **top_p** (float): Nucleus sampling (padrão: 0.85)

### Resposta

```json
{
  "audio_url": "https://storage.googleapis.com/seu-bucket/generated/joao_formal_20251222_143052_a1b2c3d4.wav?X-Goog-Algorithm=...",
  "duration": 5.32,
  "voice_id": "joao_formal",
  "language": "pt",
  "generation_time": 2.14,
  "parameters": {
    "temperature": 0.7,
    "speed": 1.0,
    "top_k": 50,
    "top_p": 0.85
  }
}
```

### Exemplo com cURL

```bash
curl -X POST https://api.runpod.ai/v2/SEU_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer SEU_API_KEY" \
  -d '{
    "input": {
      "gen_text": "Olá, mundo!",
      "ref_audio_url": "gs://meu-bucket/vozes/maria.wav",
      "voice_id": "maria_news",
      "language": "pt",
      "temperature": 0.75,
      "speed": 1.1
    }
  }'
```

### Exemplo com Python

```python
import requests

url = "https://api.runpod.ai/v2/SEU_ENDPOINT_ID/runsync"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer SEU_API_KEY"
}

payload = {
    "input": {
        "gen_text": "Este é um teste de síntese de voz.",
        "ref_audio_url": "gs://meu-bucket/vozes/carlos.wav",
        "voice_id": "carlos_podcast",
        "language": "pt",
        "temperature": 0.8,
        "speed": 0.95
    }
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()

print(f"Áudio gerado: {result['output']['audio_url']}")
print(f"Duração: {result['output']['duration']}s")
```

## 🎯 Cache de Áudios de Referência

O sistema implementa um cache inteligente que:

1. **Armazena localmente** áudios de referência em `/tmp/audio_cache/`
2. **Identifica** áudios pelo `voice_id`
3. **Reutiliza** automaticamente em próximas requisições
4. **Limpa automaticamente** quando o cache atinge 5GB
5. **Remove os mais antigos** primeiro (LRU - Least Recently Used)

### Benefícios

- ⚡ **Latência reduzida**: Evita downloads repetidos do GCS
- 💰 **Economia**: Reduz custos de egress do GCS
- 🚀 **Performance**: Inferência mais rápida

## 🔄 Código Dinâmico via GitHub

Se configurar `GITHUB_REPO_URL`, o container:

1. Clona/atualiza o repositório no startup
2. Instala dependências do `requirements.txt` (se existir)
3. Usa o `handler.py` do repositório (se existir)

**Vantagens:**
- Atualizações rápidas sem rebuild da imagem
- Iteração ágil durante desenvolvimento
- Rollback fácil via Git

## 📊 Monitoramento

O handler registra logs detalhados:

```
=== Nova solicitação ===
Texto: Olá, este é um exemplo...
Voice ID: joao_formal
Idioma: pt
Parâmetros: temp=0.7, speed=1.0, top_k=50, top_p=0.85
Áudio encontrado em cache: joao_formal
Gerando áudio com XTTS V2...
Áudio gerado em 2.14s
Fazendo upload para GCS...
Upload concluído: generated/joao_formal_20251222_143052_a1b2c3d4.wav
✓ Sucesso! Duração: 5.32s
```

## 🐛 Troubleshooting

### Erro: "GCS não configurado corretamente"

- Verifique se `GCS_CREDENTIALS` está definido
- Valide o JSON do Service Account
- Confirme que `GCS_BUCKET_NAME` está correto

### Erro: "Idioma não suportado"

- Use um dos idiomas da lista suportada
- Verifique o código do idioma (ex: `pt`, não `pt-BR`)

### Erro: "Modelo XTTS V2 não está carregado"

- Verifique logs do container durante startup
- Confirme que a GPU está disponível
- Aumente o timeout de startup no RunPod

### Cache não está funcionando

- Verifique permissões em `/tmp/audio_cache/`
- Confirme que `voice_id` é consistente entre requisições
- Monitore logs para mensagens de cache

### Upload/Download GCS falha

- Verifique permissões do Service Account
- Confirme conectividade com GCS
- Revise o formato da URL do áudio de referência

## 📝 Notas

- **Qualidade do áudio de referência**: Use áudios limpos, sem ruído, com 5-30 segundos
- **Idioma do áudio**: O áudio de referência deve estar no mesmo idioma do `gen_text`
- **URLs assinadas**: URLs do GCS expiram em 24 horas
- **Latência**: Primeira requisição por voz é mais lenta (download do GCS + cache)
- **GPU recomendada**: RTX 4090 ou superior para melhor performance

## 📄 Licença

Este projeto é fornecido como está, sem garantias. Use por sua conta e risco.

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## 🔗 Links Úteis

- [RunPod Documentation](https://docs.runpod.io/)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Google Cloud Storage](https://cloud.google.com/storage/docs)
- [XTTS V2 Model](https://huggingface.co/coqui/XTTS-v2)
