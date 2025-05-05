import os
import numpy as np
import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
import pickle
import random
import uuid
import sqlite3
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import aiofiles
import io
from datetime import datetime
import base64

# Define basic parameters
MODEL_DIR = "./models"
DB_PATH = "speakers.db"
N_MFCC = 40
N_COMPONENTS = 256
PCA_COMPONENTS = 30
THRESHOLD = 60
WEIGHT_MFCC = 0.7
WEIGHT_DELTA = 0.2
WEIGHT_DELTA_DELTA = 0.1
VAD_TOP_DB = 30
VAD_FRAME_LENGTH = 2048
VAD_HOP_LENGTH = 512

# Create directories if they don't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Initialize FastAPI app
app = FastAPI()

# Mount static files
app.mount("/", StaticFiles(directory=".", html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS audio_features
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  feature_type TEXT,
                  feature BLOB,
                  UNIQUE(name, feature_type))''')
    conn.commit()
    conn.close()

init_db()

# Spectral subtraction for noise reduction
def spectral_subtraction(audio, sr, noise_duration=0.5, n_fft=2048, hop_length=512, over_subtraction=1.0):
    try:
        noise_sample = audio[:int(noise_duration * sr)]
        if len(noise_sample) < n_fft:
            noise_sample = np.pad(noise_sample, (0, n_fft - len(noise_sample)), mode='constant')
        stft_audio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        stft_noise = librosa.stft(noise_sample, n_fft=n_fft, hop_length=hop_length)
        mag_audio, phase_audio = np.abs(stft_audio), np.angle(stft_audio)
        mag_noise = np.abs(stft_noise)
        noise_median = np.median(mag_noise, axis=1, keepdims=True)
        mag_denoised = np.maximum(mag_audio - over_subtraction * noise_median, 0.0)
        stft_denoised = mag_denoised * np.exp(1j * phase_audio)
        audio_denoised = librosa.istft(stft_denoised, hop_length=hop_length, length=len(audio))
        return audio_denoised
    except Exception as e:
        print(f"Error in spectral subtraction: {e}")
        return audio

# Extract features
def extract_features(audio=None, sr=None, n_mfcc=40, augment=True):
    try:
        y = audio
        noise_level = np.max(librosa.feature.rms(y=y))
        dynamic_top_db = max(20, min(40, 30 + 10 * np.log10(noise_level)))
        non_silent_intervals = librosa.effects.split(
            y, top_db=dynamic_top_db, frame_length=2048, hop_length=512
        )
        if len(non_silent_intervals) == 0:
            print("Warning: No speech detected after VAD, using entire audio.")
            y_speech = y
        else:
            y_speech = np.concatenate([y[start:end] for start, end in non_silent_intervals])
        y_speech = spectral_subtraction(
            y_speech, sr, noise_duration=0.5, n_fft=2048, hop_length=512, over_subtraction=1.2
        )
        augmented_features = []
        if augment:
            y_original = y_speech
            pitch_shift = random.uniform(-2, 2)
            y_pitch = librosa.effects.pitch_shift(y_speech, sr=sr, n_steps=pitch_shift)
            time_stretch = random.uniform(0.8, 1.2)
            y_stretch = librosa.effects.time_stretch(y_speech, rate=time_stretch)
            for audio_var in [y_original, y_pitch, y_stretch]:
                audio_var = librosa.effects.preemphasis(audio_var)
                mfccs = librosa.feature.mfcc(y=audio_var, sr=sr, n_mfcc=n_mfcc)
                delta_mfccs = librosa.feature.delta(mfccs)
                delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)
                augmented_features.append((mfccs.T, delta_mfccs.T, delta_delta_mfccs.T))
            mfccs = np.vstack([feat[0] for feat in augmented_features])
            delta_mfccs = np.vstack([feat[1] for feat in augmented_features])
            delta_delta_mfccs = np.vstack([feat[2] for feat in augmented_features])
        else:
            y_speech = librosa.effects.preemphasis(y_speech)
            mfccs = librosa.feature.mfcc(y=y_speech, sr=sr, n_mfcc=n_mfcc)
            delta_mfccs = librosa.feature.delta(mfccs)
            delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)
            mfccs, delta_mfccs, delta_delta_mfccs = mfccs.T, delta_mfccs.T, delta_delta_mfccs.T
        return mfccs, delta_mfccs, delta_delta_mfccs
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, None, None

# Apply CMVN
def apply_cmvn(features):
    features = np.asarray(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    normalized = (features - mean) / (std + 1e-6)
    return normalized

# Save models
def save_models(ubm, pca, gmm_models):
    ubm_path = os.path.join(MODEL_DIR, "ubm.pkl")
    pca_path = os.path.join(MODEL_DIR, "pca.pkl")
    gmm_path = os.path.join(MODEL_DIR, "gmm_models.pkl")
    with open(ubm_path, 'wb') as f:
        pickle.dump(ubm, f)
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    with open(gmm_path, 'wb') as f:
        pickle.dump(gmm_models, f)

# Load models
def load_models():
    ubm_path = os.path.join(MODEL_DIR, "ubm.pkl")
    pca_path = os.path.join(MODEL_DIR, "pca.pkl")
    gmm_path = os.path.join(MODEL_DIR, "gmm_models.pkl")
    try:
        with open(ubm_path, 'rb') as f:
            ubm = pickle.load(f)
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        with open(gmm_path, 'rb') as f:
            gmm_models = pickle.load(f)
        return ubm, pca, gmm_models
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return None, None, None

# Save features to SQLite
def save_features_to_db(speaker_id, mfccs, delta_mfccs, delta_delta_mfccs):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO audio_features (name, feature_type, feature) VALUES (?, ?, ?)",
                  (speaker_id, "mfcc", pickle.dumps(mfccs)))
        c.execute("INSERT INTO audio_features (name, feature_type, feature) VALUES (?, ?, ?)",
                  (speaker_id, "delta", pickle.dumps(delta_mfccs)))
        c.execute("INSERT INTO audio_features (name, feature_type, feature) VALUES (?, ?, ?)",
                  (speaker_id, "delta_delta", pickle.dumps(delta_delta_mfccs)))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        raise HTTPException(status_code=400, detail="Speaker ID already exists")
    finally:
        conn.close()

# Load features from SQLite
def load_features_from_db(speaker_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT feature_type, feature FROM audio_features WHERE name = ?", (speaker_id,))
    rows = c.fetchall()
    conn.close()
    features = {}
    for row in rows:
        feature_type, feature_blob = row
        features[feature_type] = pickle.loads(feature_blob)
    return (features.get("mfcc"), features.get("delta"), features.get("delta_delta"))

# Delete speaker from SQLite
def delete_speaker_from_db(speaker_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM audio_features WHERE name = ?", (speaker_id,))
    conn.commit()
    conn.close()

# Register new speaker
@app.post("/register")
async def register_speaker_endpoint(speaker_id: str = Form(...), audio: UploadFile = File(...)):
    ubm, pca, gmm_models = load_models()
    if not ubm or not pca or not gmm_models:
        raise HTTPException(status_code=500, detail="Models not loaded. Please train models first.")
    
    if speaker_id in gmm_models:
        raise HTTPException(status_code=400, detail=f"Speaker ID '{speaker_id}' already exists.")
    
    # Read audio file
    content = await audio.read()
    audio_data, sr = librosa.load(io.BytesIO(content), sr=None)
    audio_data = audio_data.astype(float) / 32768.0
    
    # Extract features
    mfccs, delta_mfccs, delta_delta_mfccs = extract_features(audio=audio_data, sr=sr, augment=True)
    if mfccs is None:
        raise HTTPException(status_code=500, detail="Failed to extract features for registration.")
    
    # Save features to database
    save_features_to_db(speaker_id, mfccs, delta_mfccs, delta_delta_mfccs)
    
    # Train GMM
    mfccs = apply_cmvn(mfccs)
    delta_mfccs = apply_cmvn(delta_mfccs)
    delta_delta_mfccs = apply_cmvn(delta_delta_mfccs)
    weighted_mfccs = WEIGHT_MFCC * mfccs
    weighted_delta = WEIGHT_DELTA * delta_mfccs
    weighted_delta_delta = WEIGHT_DELTA_DELTA * delta_delta_mfccs
    combined_features = np.hstack((weighted_mfccs, weighted_delta, weighted_delta_delta))
    combined_features_pca = pca.transform(combined_features)
    
    gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type='diag', max_iter=200, random_state=42)
    gmm.means_ = ubm.means_
    gmm.covariances_ = ubm.covariances_
    gmm.weights_ = ubm.weights_
    gmm.fit(combined_features_pca)
    
    gmm_models[speaker_id] = gmm
    save_models(ubm, pca, gmm_models)
    
    return {"status": "success", "message": f"Speaker {speaker_id} registered successfully."}

# Identify speaker
@app.post("/identify")
async def identify_speaker_endpoint(audio: UploadFile = File(...)):
    ubm, pca, gmm_models = load_models()
    if not ubm or not pca or not gmm_models:
        raise HTTPException(status_code=500, detail="Models not loaded. Please train models first.")
    
    # Read audio file
    content = await audio.read()
    audio_data, sr = librosa.load(io.BytesIO(content), sr=None)
    audio_data = audio_data.astype(float) / 32768.0
    
    # Extract features
    mfccs, delta_mfccs, delta_delta_mfccs = extract_features(audio=audio_data, sr=sr, augment=False)
    if mfccs is None:
        raise HTTPException(status_code=500, detail="Failed to extract features for identification.")
    
    # Process features
    mfccs = apply_cmvn(mfccs)
    delta_mfccs = apply_cmvn(delta_mfccs)
    delta_delta_mfccs = apply_cmvn(delta_delta_mfccs)
    weighted_mfccs = WEIGHT_MFCC * mfccs
    weighted_delta = WEIGHT_DELTA * delta_mfccs
    weighted_delta_delta = WEIGHT_DELTA_DELTA * delta_delta_mfccs
    combined_features = np.hstack((weighted_mfccs, weighted_delta, weighted_delta_delta))
    combined_features_pca = pca.transform(combined_features)
    
    # Score against all models
    likelihood_ratios = {}
    for person, gmm in gmm_models.items():
        score_target = gmm.score(combined_features_pca)
        score_ubm = ubm.score(combined_features_pca)
        likelihood_ratio = score_target - score_ubm
        likelihood_ratios[person] = likelihood_ratio
    
    min_similarity = -10
    max_similarity = 10
    percentages = {person: max(0, min(100, 100 * (lr - min_similarity) / (max_similarity - min_similarity))) 
                   for person, lr in likelihood_ratios.items()}
    
    results = [{"speaker_id": person, "similarity": round(perc, 2)} for person, perc in percentages.items()]
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    best_speaker = results[0]["speaker_id"] if results else "Unknown"
    best_percentage = results[0]["similarity"] if results else 0
    
    return {
        "status": "success",
        "best_speaker": best_speaker,
        "best_similarity": best_percentage,
        "results": results
    }

# Verify speaker
@app.post("/verify")
async def verify_speaker_endpoint(speaker_id: str = Form(...), audio: UploadFile = File(...)):
    ubm, pca, gmm_models = load_models()
    if not ubm or not pca or not gmm_models:
        raise HTTPException(status_code=500, detail="Models not loaded. Please train models first.")
    
    if speaker_id not in gmm_models:
        raise HTTPException(status_code=400, detail=f"Speaker ID '{speaker_id}' not registered.")
    
    # Read audio file
    content = await audio.read()
    audio_data, sr = librosa.load(io.BytesIO(content), sr=None)
    audio_data = audio_data.astype(float) / 32768.0
    
    # Extract features
    mfccs, delta_mfccs, delta_delta_mfccs = extract_features(audio=audio_data, sr=sr, augment=False)
    if mfccs is None:
        raise HTTPException(status_code=500, detail="Failed to extract features for verification.")
    
    # Process features
    mfccs = apply_cmvn(mfccs)
    delta_mfccs = apply_cmvn(delta_mfccs)
    delta_delta_mfccs = apply_cmvn(delta_delta_mfccs)
    weighted_mfccs = WEIGHT_MFCC * mfccs
    weighted_delta = WEIGHT_DELTA * delta_mfccs
    weighted_delta_delta = WEIGHT_DELTA_DELTA * delta_delta_mfccs
    combined_features = np.hstack((weighted_mfccs, weighted_delta, weighted_delta_delta))
    combined_features_pca = pca.transform(combined_features)
    
    # Score
    gmm = gmm_models[speaker_id]
    score_target = gmm.score(combined_features_pca)
    score_ubm = ubm.score(combined_features_pca)
    likelihood_ratio = score_target - score_ubm
    min_similarity = -10
    max_similarity = 10
    percentage = max(0, min(100, 100 * (likelihood_ratio - min_similarity) / (max_similarity - min_similarity)))
    
    return {
        "status": "success",
        "verified": percentage >= THRESHOLD,
        "similarity": round(percentage, 2),
        "speaker_id": speaker_id
    }

# Delete speaker
@app.delete("/delete_speaker/{speaker_id}")
async def delete_speaker_endpoint(speaker_id: str):
    ubm, pca, gmm_models = load_models()
    if not ubm or not pca or not gmm_models:
        raise HTTPException(status_code=500, detail="Models not loaded. Please train models first.")
    
    if speaker_id not in gmm_models:
        raise HTTPException(status_code=400, detail=f"Speaker ID '{speaker_id}' not found.")
    
    delete_speaker_from_db(speaker_id)
    del gmm_models[speaker_id]
    save_models(ubm, pca, gmm_models)
    
    return {"status": "success", "message": f"Speaker {speaker_id} deleted successfully."}

# Get all speakers
@app.get("/speakers")
async def get_speakers_endpoint():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT DISTINCT name FROM audio_features")
    speakers = [row[0] for row in c.fetchall()]
    conn.close()
    return {"status": "success", "speakers": speakers}