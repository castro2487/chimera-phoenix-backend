import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # Carga .env si existe (opcional)
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/TheBloke/DeepSeek-Coder-V2-Lite-GGUF"

class Prompt(BaseModel):
    prompt: str

app = FastAPI()

@app.get("/")
def health():
    return {"status": "Phoenix proxy alive 🚀"}

@app.post("/chat")
async def chat_endpoint(data: Prompt):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    resp = requests.post(API_URL, headers=headers, json={"inputs": data.prompt}, timeout=120)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    out = resp.json()
    # Ajusta según formato de salida
    if isinstance(out, list) and "generated_text" in out[0]:
        text = out[0]["generated_text"]
    else:
        text = out.get("generated_text", str(out))
    return {"response": text}
