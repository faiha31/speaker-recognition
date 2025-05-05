from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pickle
import aiofiles

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
def init_db():
    conn = sqlite3.connect("speakers.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS speakers (
        speaker_id TEXT PRIMARY KEY,
        audio_path TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

# Model loading (if available)
ubm_model = None
pca = None
gmm_models = {}
try:
    ubm_model = pickle.load(open("models/ubm.pkl", "rb"))
    pca = pickle.load(open("models/pca.pkl", "rb"))
    gmm_models = pickle.load(open("models/gmm_models.pkl", "rb"))
except FileNotFoundError:
    print("Models not loaded. Identification and verification will not work until models are trained.")

# Register endpoint
@app.post("/register")
async def register_speaker_endpoint(speaker_id: str = Form(...), audio: UploadFile = File(...)):
    if not audio.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    # Save audio file
    audio_path = f"audio/{speaker_id}.wav"
    os.makedirs("audio", exist_ok=True)
    async with aiofiles.open(audio_path, 'wb') as out_file:
        content = await audio.read()
        await out_file.write(content)
    
    # Store in database
    conn = sqlite3.connect("speakers.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO speakers (speaker_id, audio_path) VALUES (?, ?)", (speaker_id, audio_path))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Speaker ID already exists")
    conn.close()
    
    return {"status": "success", "message": f"Speaker {speaker_id} registered successfully"}

# Get all speakers
@app.get("/speakers")
async def get_speakers():
    conn = sqlite3.connect("speakers.db")
    c = conn.cursor()
    c.execute("SELECT speaker_id FROM speakers")
    speakers = [row[0] for row in c.fetchall()]
    conn.close()
    return {"status": "success", "speakers": speakers}

# Delete speaker
@app.delete("/delete_speaker/{speaker_id}")
async def delete_speaker_endpoint(speaker_id: str):
    conn = sqlite3.connect("speakers.db")
    c = conn.cursor()
    c.execute("SELECT audio_path FROM speakers WHERE speaker_id = ?", (speaker_id,))
    result = c.fetchone()
    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Speaker not found")
    
    audio_path = result[0]
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    c.execute("DELETE FROM speakers WHERE speaker_id = ?", (speaker_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"Speaker {speaker_id} deleted successfully"}

# Placeholder for identify and verify (requires models)
@app.post("/identify")
async def identify_speaker_endpoint(audio: UploadFile = File(...)):
    if not ubm_model or not pca or not gmm_models:
        raise HTTPException(status_code=400, detail="Models not loaded")
    # Implementation omitted for brevity
    return {"status": "success", "best_speaker": "SPK_001", "best_similarity": 95.0}

@app.post("/verify")
async def verify_speaker_endpoint(speaker_id: str = Form(...), audio: UploadFile = File(...)):
    if not ubm_model or not pca or not gmm_models:
        raise HTTPException(status_code=400, detail="Models not loaded")
    # Implementation omitted for brevity
    return {"status": "success", "verified": True, "similarity": 95.0}

# Test endpoint for debugging
@app.post("/debug_post")
async def debug_post():
    return {"status": "success", "message": "POST request received"}
