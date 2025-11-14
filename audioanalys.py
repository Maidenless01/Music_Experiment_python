# This is SCRIPT 1: The Audio Analyzer
#
# It reads an audio file, analyzes it for visualization data,
# and saves that data to a compressed file (`analysis_data.npz`).
#
# Requires: pip install librosa numpy
#
# Run this script ONCE for each song you want to visualize.

import librosa
import numpy as np
import sys

# --- Configuration ---
# Your original audio file
AUDIO_FILE = "DAF.wav"
# The file to save the analysis data to
OUTPUT_DATA_FILE = "analysis_data.npz" 

# Analysis parameters
SR = 44100          # Sample Rate to use. 
HOP_LENGTH = 512    # The "step size" for analysis frames. Smaller = smoother but more data.
N_MELS = 128        # Number of frequency bins to create. This will be our "NUM_BARS".

def analyze_audio():
    print(f"Loading audio file: {AUDIO_FILE}...")
    try:
        # Load the audio file
        # y = audio time series, sr = sample rate
        y, sr = librosa.load(AUDIO_FILE, sr=SR)
    except FileNotFoundError:
        print(f"Error: File '{AUDIO_FILE}' not found.")
        print("Please place the audio file in the same directory.")
        sys.exit()
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit()

    print("Analyzing... (This may take a moment)")

    # 1. Calculate the Mel Spectrogram (Frequency data)
    # This is like running the FFT on the whole song.
    # The result is a 2D array: (N_MELS, T)
    # T = number of time frames
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    
    # 2. Calculate RMS (Volume)
    # This gives us the "loudness" for each time frame.
    # Result is a 2D array: (1, T)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    
    # 3. Detect Beat Onsets (Beat detection)
    # This finds the *time frames* where beats are likely to occur.
    # Result is a 1D array of frame indices.
    beat_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH, units='frames')

    # 4. Calculate the Visualizer's Playback FPS
    # This is crucial for syncing the visuals with the audio.
    # It's based on how we "stepped" through the audio (hop_length).
    vis_fps = sr / HOP_LENGTH

    print(f"Analysis complete.")
    print(f"  - Spectrogram shape: {mel_spec.shape}")
    print(f"  - RMS shape: {rms.shape}")
    print(f"  - Beats found: {len(beat_frames)}")
    print(f"  - Visualizer FPS: {vis_fps:.2f}")

    # 5. Save all our data to a single compressed file
    try:
        np.savez_compressed(
            OUTPUT_DATA_FILE,
            mel_spec=mel_spec,
            rms=rms,
            beat_frames=beat_frames,
            vis_fps=vis_fps,
            audio_file=AUDIO_FILE # Store the original audio file name
        )
        print(f"Analysis data saved to {OUTPUT_DATA_FILE}")
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    analyze_audio()