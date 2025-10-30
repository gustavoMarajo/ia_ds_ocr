from fastapi import FastAPI, UploadFile, File
from transformers import AutoModel, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import torch, io, warnings, logging, os, asyncio, tempfile, shutil, time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from glob import glob


# ---------------------------------------------------------------------
# SUPRESSÃO DE WARNINGS E LOGS
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# ---------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------
app = FastAPI(title="DeepSeek-OCR API", version="1.0")

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🔹 Carregando modelo na inicialização...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code = True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, use_safetensors = True)
model = model.eval().to(torch.bfloat16 if DEVICE == "cuda" else torch.float32).to(DEVICE)

# Executor de threads — permite múltiplos OCRs simultâneos
executor = ThreadPoolExecutor(max_workers = os.cpu_count())

# cria diretório temporário fixo
BASE_TEMP_DIR = "./temp_outputs"
os.makedirs(BASE_TEMP_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# FUNÇÃO AUXILIAR PARA LIMPAZA DO DIRETÓRIO TEMPORÁRIO
# ---------------------------------------------------------------------
def clean_temp_dir():
    if os.path.exists(BASE_TEMP_DIR):
        shutil.rmtree(BASE_TEMP_DIR)
    os.makedirs(BASE_TEMP_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# FUNÇÃO OCR
# ---------------------------------------------------------------------
def run_ocr(image_bytes: bytes) -> str:

    start = time.time()
    print(f"🟡 Iniciando OCR em {os.getpid()} - {start:.2f}")

    # pasta exclusiva por requisição
    req_id = str(uuid4())
    req_dir = os.path.join(BASE_TEMP_DIR, req_id)
    os.makedirs(req_dir, exist_ok=True)

    # salva a imagem da requisição
    img_path = os.path.join(req_dir, "input.jpg")
    with open(img_path, "wb") as f:
        f.write(image_bytes)

    # pasta de saída para o infer
    out_dir = os.path.join(req_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

    prompt = "<image>\nFree OCR."

    print(f"🔹 Rodando OCR em {img_path}")
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

    # procura texto nos .txt
    print("📁 Buscando .txt em:", out_dir)
    txts = sorted(glob(os.path.join(out_dir, "**", "*.mmd"), recursive=True), key=os.path.getmtime, reverse=True)
    if not txts:
        txts = sorted(glob(os.path.join(req_dir, "**", "*.mmd"), recursive=True), key=os.path.getmtime, reverse=True)
    if not txts:
        txts = sorted(glob(os.path.join(BASE_TEMP_DIR, "**", "*.mmd"), recursive=True), key=os.path.getmtime, reverse=True)

    if not txts:
        print("⚠️ Nenhum arquivo .txt encontrado!")
        return ""

    latest_txt = txts[0]
    print("🧾 Lendo:", latest_txt)
    with open(latest_txt, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()

    try:
        shutil.rmtree(req_dir)
        print(f"🧹 Limpeza: diretório {req_dir} removido com sucesso.")
    except Exception as e:
        print(f"⚠️ Falha ao remover {req_dir}: {e}")

    print(f"🟢 Finalizado em {time.time()-start:.2f}s")
    
    return text


#********************************************************************************************************************************************       
# ---------------------------------------------------------------------
# ENDPOINT POST /ocr
# ---------------------------------------------------------------------
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # limpa antes de processar, não depois
    #clean_temp_dir()

    # Executa OCR em thread separada para não travar o loop principal
    text = await asyncio.get_running_loop().run_in_executor(executor, run_ocr, image_bytes)
    return {"filename": file.filename, "text": text}
