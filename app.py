import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

class Prompt(BaseModel):
    prompt: str

app = FastAPI()

# Descargar y cargar el modelo
model_path = hf_hub_download(
    repo_id="ijohn07/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M-GGUF",
    filename="deepseek-coder-v2-lite-instruct-q4_k_m.gguf",
    cache_dir="/app/models"
)

llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4)

@app.get("/")
def read_root():
    return {"status": "Modelo cargado correctamente."}

@app.post("/chat")
async def chat(prompt: Prompt):
    try:
        output = llm(prompt.prompt, max_tokens=256, temperature=0.7)
        return {"response": output["choices"][0]["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
