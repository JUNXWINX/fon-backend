# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import shutil
import os
import scipy.io.wavfile
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    VitsModel
)

app = FastAPI(title="API Assistant Fon")

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")

# --- Chargement des modèles (une seule fois au démarrage) ---
print("Chargement ASR...")
asr = pipeline(
    "automatic-speech-recognition",
    model="Professor/mms-300m-fongbe",
    device=0 if device == "cuda" else -1
)

print("Chargement traduction...")
nllb_name = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_name).to(device)

print("Chargement TTS...")
tts_model = VitsModel.from_pretrained("facebook/mms-tts-fon").to(device)
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fon")
print("Tous les modèles sont prêts !")

# --- Modèles de données ---
class TextRequest(BaseModel):
    text: str

# --- Endpoints ---
@app.get("/")
def racine():
    return {"status": "API Fon opérationnelle"}

@app.post("/transcribe")
async def transcrire_audio(file: UploadFile = File(...)):
    try:
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        resultat = asr(temp_path)
        os.remove(temp_path)
        return {"text": resultat["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def traduire(request: TextRequest):
    try:
        nllb_tokenizer.src_lang = "fon_Latn"
        inputs = nllb_tokenizer(request.text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = nllb_model.generate(
                **inputs,
                forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids("fra_Latn"),
                max_length=200
            )
        traduction = nllb_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"translation": traduction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speak")
async def parler(request: TextRequest):
    try:
        inputs = tts_tokenizer(request.text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = tts_model(**inputs).waveform
        audio_path = "/tmp/reponse.wav"
        scipy.io.wavfile.write(
            audio_path,
            rate=tts_model.config.sampling_rate,
            data=output.cpu().numpy().squeeze()
        )
        return FileResponse(audio_path, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
