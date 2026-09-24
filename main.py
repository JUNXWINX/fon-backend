import os
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="API Assistant Fon")

# --- Configuration ---
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN non défini")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- URLs des modèles Hugging Face ---
# Remplace TON_USERNAME par ton pseudo Hugging Face
ASR_MODEL = "JunxWinx/fon-asr-fongbe-v2"
TRANSLATION_MODEL = "masakhane/m2m100_418M_fon_fr_rel_news"
TTS_MODEL = "facebook/mms-tts-fon"

ASR_URL = f"https://api-inference.huggingface.co/models/{ASR_MODEL}"
TRANS_URL = f"https://api-inference.huggingface.co/models/{TRANSLATION_MODEL}"
TTS_URL = f"https://api-inference.huggingface.co/models/{TTS_MODEL}"

# --- Modèles de données ---
class TextRequest(BaseModel):
    text: str

# --- Endpoints ---

@app.get("/")
def racine():
    return {"status": "API Fon opérationnelle", "models": {
        "asr": ASR_MODEL,
        "translation": TRANSLATION_MODEL,
        "tts": TTS_MODEL
    }}

@app.post("/transcribe")
async def transcrire_audio(file: UploadFile = File(...)):
    """Reçoit un audio en fon, retourne la transcription."""
    try:
        audio_data = await file.read()
        response = requests.post(ASR_URL, headers=HEADERS, data=audio_data, timeout=60)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur ASR: {str(e)}")

@app.post("/translate")
async def traduire(request: TextRequest):
    """Traduit un texte du fon vers le français."""
    try:
        payload = {
            "inputs": request.text,
            "parameters": {"src_lang": "fon", "tgt_lang": "fra"}
        }
        response = requests.post(TRANS_URL, headers=HEADERS, json=payload, timeout=60)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        result = response.json()
        # L'API retourne une liste de dicts
        if isinstance(result, list) and len(result) > 0:
            return {"translation": result[0].get("translation_text", "")}
        return result
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur traduction: {str(e)}")

@app.post("/speak")
async def parler(request: TextRequest):
    """Génère un audio à partir d'un texte en fon."""
    try:
        payload = {"inputs": request.text}
        response = requests.post(TTS_URL, headers=HEADERS, json=payload, timeout=60)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return Response(content=response.content, media_type="audio/wav")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur TTS: {str(e)}")
