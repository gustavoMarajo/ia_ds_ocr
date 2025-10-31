# Base Ubuntu + Python
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Evita prompts na instalação
ENV DEBIAN_FRONTEND=noninteractive

# Atualiza pacotes e instala dependências
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean

# Cria diretório da aplicação
WORKDIR /app

# Copia os arquivos
COPY . .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Expõe a porta da API
EXPOSE 8000

# Comando para iniciar o servidor FastAPI com Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
