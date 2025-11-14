# This is SCRIPT 1: The NEW Hybrid Analyzer
#
# It runs BOTH the librosa analysis (for beats, bpm, waves)
# AND the Transformer analysis (for AI-driven color)
# and saves it all to one file.
#
# This will be SLOW. Run it once per song.

import librosa
import numpy as np
import sys
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
AUDIO_FILE = "daf.wav" 
OUTPUT_DATA_FILE = "hybrid_analysis_data.npz" 

# --- Librosa Config ---
SR = 44100          
HOP_LENGTH = 512    
N_MELS = 128        

# --- AI Config ---
MODEL_NAME = "facebook/wav2vec2-base-960h"
AI_SAMPLE_RATE = 16000 # Wav2Vec2 expects 16kHz

def analyze_audio():
    print(f"Loading audio file: {AUDIO_FILE}...")
    try:
        y, sr = librosa.load(AUDIO_FILE, sr=SR)
        # Load a second copy for the AI model
        y_ai, sr_ai = librosa.load(AUDIO_FILE, sr=AI_SAMPLE_RATE)
    except FileNotFoundError:
        print(f"Error: File '{AUDIO_FILE}' not found.")
        sys.exit()
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit()
    
    total_duration_sec = len(y) / SR

    # --- 1. Librosa Analysis (Beats, Mel, RMS) ---
    print("Analyzing... (Part 1/2: Librosa - Beats, Waves, BPM)...")
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH, units='frames')
    avg_tempo = np.mean(tempo)
    vis_fps = sr / HOP_LENGTH
    
    # This is the total number of frames our visualizer will have
    num_vis_frames = mel_spec.shape[1]

    print("Librosa analysis complete.")
    print(f"  - Detected BPM: {avg_tempo:.2f}")
    print(f"  - Visualizer FPS: {vis_fps:.2f}")
    print(f"  - Total visual frames: {num_vis_frames}")

    # --- 2. Transformer AI Analysis (Color) ---
    print("Analyzing... (Part 2/2: AI Transformer - This is SLOW)...")
    try:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading AI model. Do you have internet? {e}")
        sys.exit()

    # Process in chunks to save memory
    chunk_duration = 10 
    chunk_samples = chunk_duration * sr_ai
    all_hidden_states = []
    
    for i in range(0, len(y_ai), chunk_samples):
        chunk = y_ai[i : i + chunk_samples]
        if len(chunk) < 1000: continue

        inputs = processor(chunk, sampling_rate=sr_ai, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state.squeeze(0).numpy()
            all_hidden_states.append(hidden_states)
            
        print(f"   ...processed AI chunk {i//chunk_samples + 1}...")

    full_embeddings = np.concatenate(all_hidden_states, axis=0)
    num_ai_frames = full_embeddings.shape[0]
    print(f"   Raw AI Data Shape: {full_embeddings.shape}") 

    print("   ...compressing AI data to RGB colors (PCA)...")
    pca = PCA(n_components=3)
    rgb_data = pca.fit_transform(full_embeddings)
    scaler = MinMaxScaler(feature_range=(0, 255))
    rgb_data = scaler.fit_transform(rgb_data).astype(int)

    print("   AI analysis complete.")

    # --- 3. CRITICAL: Sync AI data to Librosa data ---
    # We have two different datasets:
    #   - mel_spec: (128, num_vis_frames)
    #   - rgb_data: (num_ai_frames, 3)
    # We need to "stretch" the AI data to match the visualizer frames.
    
    print("Syncing AI data to visualizer timeline...")
    
    # Original AI time-points (e.g., 0 to 1800)
    original_ai_indices = np.linspace(0, num_ai_frames - 1, num_ai_frames)
    # New visualizer time-points (e.g., 0 to 5200)
    target_vis_indices = np.linspace(0, num_ai_frames - 1, num_vis_frames)

    # Use interpolation to resample R, G, and B channels
    r_channel = np.interp(target_vis_indices, original_ai_indices, rgb_data[:, 0])
    g_channel = np.interp(target_vis_indices, original_ai_indices, rgb_data[:, 1])
    b_channel = np.interp(target_vis_indices, original_ai_indices, rgb_data[:, 2])

    # Stack them back together
    synced_rgb_data = np.stack((r_channel, g_channel, b_channel), axis=1)
    
    print(f"Sync complete. Final RGB data shape: {synced_rgb_data.shape}")

    # --- 4. Save all data ---
    print(f"Saving all data to {OUTPUT_DATA_FILE}...")
    try:
        np.savez_compressed(
            OUTPUT_DATA_FILE,
            mel_spec=mel_spec,
            rms=rms,
            beat_frames=beat_frames,
            vis_fps=vis_fps,
            tempo=avg_tempo,
            audio_file=AUDIO_FILE,
            rgb_data=synced_rgb_data # ADD THE NEW AI DATA
        )
        print("Done. You can now run the fluid visualizer.")
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    analyze_audio()