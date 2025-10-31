from fastapi import FastAPI, Form, UploadFile, File
from transformers import AutoModel, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
# from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4
from glob import glob
from contextlib import redirect_stdout, redirect_stderr
import torch, io, warnings, logging, os, asyncio, shutil, time, sqlite3, numpy as np, json, sys, importlib, builtins, types, sys


# ---------------------------------------------------------------------
# SUPRESSÃO DE WARNINGS E LOGS
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# ---------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_TEMP_DIR = "./temp_outputs"
DB_PATH = "./document_types.db"
DEVNULL = open(os.devnull, "w", encoding="utf-8", buffering=1)

os.makedirs(BASE_TEMP_DIR, exist_ok=True)
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = DEVNULL
sys.stderr = DEVNULL


print("Carregando modelo na inicialização...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, use_safetensors=True)

_builtin_print = builtins.print

def safe_print(*args, **kwargs):
    """Print seguro que ignora stdout fechado."""
    try:
        _builtin_print(*args, **kwargs)
    except (ValueError, OSError):
        pass

# Substitui print global
builtins.print = safe_print

# Se o modelo DeepSeek-OCR tiver referência local ao print (interno), substitui também
for name, module in sys.modules.items():
    if name and "deepseek" in name.lower() and hasattr(module, "__dict__"):
        if "print" in module.__dict__:
            module.__dict__["print"] = safe_print



model = model.eval().to(torch.bfloat16 if DEVICE == "cuda" else torch.float32).to(DEVICE)

app = FastAPI(title="DeepSeek-OCR API", version="1.0")

# Executor de threads — modelo é compartilhado entre threads
executor = ThreadPoolExecutor(max_workers=os.cpu_count())


# ---------------------------------------------------------------------
# FUNÇÃO OCR (usa modelo global)
# ---------------------------------------------------------------------
def run_ocr(image_bytes: bytes) -> str:
    req_id = str(uuid4())
    req_dir = os.path.join(BASE_TEMP_DIR, req_id)
    os.makedirs(req_dir, exist_ok=True)

    img_path = os.path.join(req_dir, "input.jpg")
    with open(img_path, "wb") as f:
        f.write(image_bytes)

    out_dir = os.path.join(req_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

    prompt = "<image>\nFree OCR."

    # redireciona stdout/stderr para evitar logs do modelo
    # with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
    _ = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=img_path,
        output_path=out_dir,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,
        test_compress=True
    )

    txts = sorted(glob(os.path.join(out_dir, "**", "*.mmd"), recursive=True), key=os.path.getmtime, reverse=True)
    if not txts:
        txts = sorted(glob(os.path.join(req_dir, "**", "*.mmd"), recursive=True), key=os.path.getmtime, reverse=True)

    if not txts:
        shutil.rmtree(req_dir, ignore_errors=True)
        return ""

    with open(txts[0], "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()

    shutil.rmtree(req_dir, ignore_errors=True)
    return text


# ---------------------------------------------------------------------
# FUNÇÕES AUXILIARES DE BANCO DE DADOS (SQLite)
# ---------------------------------------------------------------------
def init_db():
    """Cria tabela se não existir"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS document_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_embedding(tipo: str, embedding: np.ndarray):
    """Salva embedding e tipo no banco"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    emb_json = json.dumps(embedding.tolist())
    c.execute("INSERT INTO document_embeddings (tipo, embedding) VALUES (?, ?)", (tipo, emb_json))
    conn.commit()
    conn.close()

def load_all_embeddings():
    """Carrega todos embeddings e tipos"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT tipo, embedding FROM document_embeddings")
    rows = c.fetchall()
    conn.close()
    tipos = []
    embeddings = []
    for tipo, emb_json in rows:
        tipos.append(tipo)
        embeddings.append(np.array(json.loads(emb_json)))
    return tipos, np.array(embeddings) if embeddings else ([], np.empty((0, 1)))


# ---------------------------------------------------------------------
# FUNÇÃO PARA EXTRAIR EMBEDDING DE UMA IMAGEM (SEM OCR)
# ---------------------------------------------------------------------
def extract_image_embedding(image_bytes: bytes) -> np.ndarray:
    """Extrai vetor de características (embedding) da imagem"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # O método infer do modelo faz OCR, então usamos encode visual
    # Para DeepSeek-OCR, o visual encoder fica dentro do modelo principal
    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            inputs = tokenizer(images=image, text="<image>\n", return_tensors="pt").to(DEVICE)
            emb = model.get_image_features(**inputs)
        else:
            # fallback simples
            inputs = tokenizer(images=image, text="<image>\n", return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)

        emb = emb.detach().cpu().numpy().flatten()
    return emb


# ---------------------------------------------------------------------
# ENDPOINT POST /ocr
# ---------------------------------------------------------------------
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    # async with ocr_lock:
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(executor, run_ocr, image_bytes)
    return {"filename": file.filename, "text": text}

# ---------------------------------------------------------------------
# ENDPOINTs POST /treinamento e /predição de tipo de documento
# ---------------------------------------------------------------------
# @app.post("/train-document-type")
# async def train_document_type(file: UploadFile = File(...), tipo: str = Form(...)):
#     image_bytes = await file.read()
#     emb = extract_image_embedding(image_bytes)
#     save_embedding(tipo, emb)
#     return {"status": "ok", "message": f"Documento de tipo '{tipo}' treinado com sucesso."}


# @app.post("/predict-document-type")
# async def predict_document_type(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     emb = extract_image_embedding(image_bytes)

#     tipos, banco = load_all_embeddings()
#     if not tipos:
#         return {"error": "Nenhum documento treinado ainda."}

#     sims = cosine_similarity([emb], banco)
#     idx = int(np.argmax(sims))
#     return {
#         "tipo_predito": tipos[idx],
#         "similaridade": float(sims[0][idx])
#     }
