import numpy as np
import librosa
import soundfile as sf
import os
import glob
from pathlib import Path

# ==============================
# CONFIG (you can move to config.py later)
# ==============================
SAMPLE_RATE = 16000
DURATION = 3  # seconds
N_MFCC = 40

# Data directories
RAW_VOICE_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/voice")
PROCESSED_VOICE_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed/voice")


# ==============================
# MAIN PREPROCESS FUNCTION
# ==============================
def preprocess_audio(file_path):
    try:
        # 🔹 Load audio
        data, samplerate = sf.read(file_path)

        # 🔹 Convert to mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # 🔹 Convert to float32
        data = data.astype(np.float32)

        # 🔹 Normalize audio
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))

        # 🔹 Resample to 16kHz
        if samplerate != SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=samplerate, target_sr=SAMPLE_RATE)

        # 🔹 Fix audio length (important for model consistency)
        max_length = SAMPLE_RATE * DURATION
        if len(data) > max_length:
            data = data[:max_length]
        else:
            data = np.pad(data, (0, max_length - len(data)))

        # 🔹 Trim silence (optional - can be slow on some systems)
        try:
            data, _ = librosa.effects.trim(data)
        except Exception as e:
            print(f"Warning: Could not trim audio: {e}")
            # Continue without trimming

        # ==============================
        # FEATURE EXTRACTION
        # ==============================

        # 🔸 MFCC (main feature)
        mfcc = librosa.feature.mfcc(y=data, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

        # 🔸 Chroma
        chroma = librosa.feature.chroma_stft(y=data, sr=SAMPLE_RATE)

        # 🔸 Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=data, sr=SAMPLE_RATE)

        # 🔹 Take mean (convert to 1D)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        chroma_mean = np.mean(chroma.T, axis=0)
        mel_mean = np.mean(mel.T, axis=0)

        # 🔹 Combine features
        features = np.hstack([mfcc_mean, chroma_mean, mel_mean])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ==============================
# OPTIONAL: TEST FUNCTION
# ==============================
def create_test_audio(file_path, duration=3, sample_rate=16000):
    """Generate a synthetic test audio file for testing."""
    # Create synthetic audio (sine wave)
    t = np.linspace(0, duration, int(sample_rate * duration))
    # 440 Hz sine wave (A4 note)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(file_path, audio, sample_rate)
    print(f"Created test audio file: {file_path}")


def find_voice_files(root_dir):
    """Recursively find all WAV files in the directory."""
    wav_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                wav_files.append(os.path.join(dirpath, filename))
    return sorted(wav_files)


def batch_preprocess_voice_files(input_dir, output_dir=None, max_files=None):
    """Preprocess all voice files in a directory."""
    # Find all WAV files
    wav_files = find_voice_files(input_dir)
    
    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return []
    
    if max_files:
        wav_files = wav_files[:max_files]
    
    print(f"Found {len(wav_files)} voice files. Processing...")
    
    results = []
    failed_files = []
    
    for i, file_path in enumerate(wav_files, 1):
        relative_path = os.path.relpath(file_path, input_dir)
        print(f"[{i}/{len(wav_files)}] Processing: {relative_path}...", end=" ")
        
        features = preprocess_audio(file_path)
        
        if features is not None:
            result = {
                'file': file_path,
                'relative_path': relative_path,
                'features_shape': features.shape,
                'features': features
            }
            results.append(result)
            print(f"✓ Shape: {features.shape}")
        else:
            failed_files.append(file_path)
            print("✗ Failed")
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed: {len(results)}/{len(wav_files)}")
    print(f"Failed: {len(failed_files)}/{len(wav_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {os.path.relpath(f, input_dir)}")
    
    return results


if __name__ == "__main__":
    # Process all voice files from data/raw/voice (no limit)
    print("Starting full preprocessing of all voice files...")
    print(f"Input directory: {RAW_VOICE_DIR}\n")
    
    results = batch_preprocess_voice_files(RAW_VOICE_DIR)
    
    if results:
        print(f"\n{'='*50}")
        print(f"✓ Preprocessing complete!")
        print(f"✓ Total files processed: {len(results)}")
        print(f"{'='*50}")