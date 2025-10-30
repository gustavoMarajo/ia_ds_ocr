from fastapi import FastAPI, UploadFile, File
from transformers import AutoModel, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import torch, io, warnings, logging, os, asyncio, shutil, time
from uuid import uuid4
from glob import glob
from contextlib import redirect_stdout, redirect_stderr

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

print("🔹 Carregando modelo na inicialização...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, use_safetensors=True)
model = model.eval().to(torch.bfloat16 if DEVICE == "cuda" else torch.float32).to(DEVICE)

app = FastAPI(title="DeepSeek-OCR API", version="1.0")

# Executor de threads — modelo é compartilhado entre threads
executor = ThreadPoolExecutor(max_workers=10)

BASE_TEMP_DIR = "./temp_outputs"
os.makedirs(BASE_TEMP_DIR, exist_ok=True)


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
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
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
# ENDPOINT POST /ocr
# ---------------------------------------------------------------------
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(executor, run_ocr, image_bytes)
    return {"filename": file.filename, "text": text}
