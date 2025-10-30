## INSTRUÇÕES DE USO

## 🔹Dependencias
Instalar os packages e seguir instruções de uso em "requiriments.txt"

## 🔹Inicialização do serviço - API
```
- uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 
```

## 🔹Captura de dependencias - estado atual
```
- pip freeze > requirements_freeze.txt
```

## 🔹Modelo IA
O modelo DeepSeek-OCR é baixado automaticamente na primeira execução
##

Se estiver usando GPU com CUDA 11.8:
```
- pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

Se estiver usando apenas CPU:
```
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 🔹Projeto
- Python 3.10+ (Windows 11)
- DeepSeek-OCR (via transformers, trust_remote_code=True)
- Execução em API FastAPI com suporte a uploads e OCR em GPU ou CPU.