# Imagem base enxuta
FROM python:3.12-slim

# Evita gerar .pyc e força flush de logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Diretório de trabalho
WORKDIR /app

# Instala dependências do sistema (caso xgboost precise)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements primeiro (melhora cache de build)
COPY requirements-deploy.txt .

# Instala dependências
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copia somente src
COPY src ./src

# Expõe a porta (Render normalmente usa 10000)
EXPOSE 10000

# Comando para iniciar API
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]