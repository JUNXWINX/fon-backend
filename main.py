# main.py
import os
import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

app = FastAPI(title="API Assistant Fon")

# Configuration de l'appareil (CPU sur ClawCloud)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation de l'appareil : {device}")

# Chargement du modèle fine-tuné
MODEL_PATH = "/app/model/fon-asr-final-v2"
print("Chargement du modèle ASR...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).to(device)
print("Modèle chargé avec succès !")

# Modèle pour les requêtes texte
class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "API Fon opérationnelle"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Sauvegarde temporaire du fichier audio
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Chargement et transcription
        audio, sr = sf.read(temp_path)
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]
        
        os.remove(temp_path)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
