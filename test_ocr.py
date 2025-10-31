from transformers import AutoModel, AutoTokenizer
import torch
import os
import warnings
import logging

# ---------------------------------------------------------------------
# REMOÇÃO DE AVISOS DESNECESSÁRIOS
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# CONFIGURAÇÃO DE LOGS - Para garantir uma saída limpa no terminal
# ---------------------------------------------------------------------
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# ---------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use a GPU, se houver
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

# ---------------------------------------------------------------------
# CARREGAR MODELO E TOKENIZER
# ---------------------------------------------------------------------
print("Carregando modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    #_attn_implementation="flash_attention_2",  # usa flash-attn se disponível
    use_safetensors=True
)
model = model.eval().to(torch.bfloat16).cuda()  # usa GPU (BF16)

# ---------------------------------------------------------------------
# PARAMETROS DE OCR
# ---------------------------------------------------------------------
image_file = "C:/Users/Gustavo/\Documents/Documentos de Treinamento CX/Nova pasta (1)/Nova pasta/215c87bd-92ce-46e0-901c-78a138ad500f.jpg"        # sua imagem
output_path = "./results"           # pasta de saída (cria se não existir)
prompt = "<image>\nFree OCR."       # prompt básico do modelo

os.makedirs(output_path, exist_ok=True)

# ---------------------------------------------------------------------
# EXECUTAR OCR (usando método customizado do DeepSeek)
# ---------------------------------------------------------------------
print("Rodando inferência...")
res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,     # salva imagens + textos gerados
    test_compress=True     # usa o modo de compressão contextual
)

print("OCR finalizado!")
print("Resultado salvo em:", output_path)
