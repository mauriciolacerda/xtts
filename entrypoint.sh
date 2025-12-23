#!/bin/bash
set -e

echo "=== XTTS V2 RunPod Entrypoint ==="

# Verificar se GITHUB_REPO_URL está definido
if [ -z "$GITHUB_REPO_URL" ]; then
    echo "WARNING: GITHUB_REPO_URL não está definido. Pulando git clone/pull."
else
    echo "Repositório GitHub: $GITHUB_REPO_URL"
    
    # Diretório do repositório
    REPO_DIR="/app/repo"
    
    # Verificar se o repositório já foi clonado
    if [ -d "$REPO_DIR/.git" ]; then
        echo "Repositório já existe. Atualizando..."
        cd $REPO_DIR
        git pull origin main || git pull origin master || echo "Pull falhou, continuando com código existente"
    else
        echo "Clonando repositório..."
        git clone $GITHUB_REPO_URL $REPO_DIR
        cd $REPO_DIR
    fi
    
    # Instalar dependências extras se requirements.txt existir
    if [ -f "$REPO_DIR/requirements.txt" ]; then
        echo "Instalando dependências extras do repositório..."
        pip install -r requirements.txt
    fi
    
    # Verificar se handler.py existe no repositório
    if [ -f "$REPO_DIR/handler.py" ]; then
        echo "Usando handler.py do repositório..."
        exec python -u $REPO_DIR/handler.py
    else
        echo "handler.py não encontrado no repositório. Usando handler padrão..."
    fi
fi

# Se chegou aqui, usar handler.py local (se existir)
if [ -f "/app/handler.py" ]; then
    echo "Iniciando handler.py local..."
    exec python -u /app/handler.py
else
    echo "ERRO: Nenhum handler.py encontrado!"
    exit 1
fi
