import os
import numpy as np
import pyaudio
import librosa
import tensorflow as tf
import math

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "rafeeq_model.tflite"
LABELS_PATH = "labels.txt"

SAMPLE_RATE = 16000
DURATION = 1.5
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
CHUNK = 1024

# ⚠️ TUNE THIS: If it triggers on background noise, increase this number (e.g., 0.05)
# If it ignores your voice, decrease it (e.g., 0.01)
VOLUME_THRESHOLD = 0.02  

# ==========================================
# 2. LOAD MODEL & LABELS
# ==========================================
print("📦 Loading Custom Rafeeq AI...")
# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels into a list
with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
print(f"✅ Loaded {len(labels)} commands: {labels}")

# ==========================================
# 3. FEATURE EXTRACTION (Must match Colab exactly)
# ==========================================
def extract_features(audio_array):
    """Processes live audio exactly how the training data was processed."""
    # Trim silence
    y_trimmed, _ = librosa.effects.trim(audio_array, top_db=20)
    
    # Pad or truncate to exact 1.5 second length
    if len(y_trimmed) > SAMPLES_PER_TRACK:
        y_trimmed = y_trimmed[:SAMPLES_PER_TRACK]
    else:
        padding = SAMPLES_PER_TRACK - len(y_trimmed)
        y_trimmed = np.pad(y_trimmed, (0, padding), 'constant')
        
    # Extract MFCC
    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=SAMPLE_RATE, n_mfcc=13, n_fft=2048, hop_length=512)
    
    # Reshape for the CNN: (Batch=1, TimeSteps, MFCCs, Channels=1)
    # The expected shape is likely (1, 47, 13, 1) depending on librosa version
    mfccs = mfccs[np.newaxis, ..., np.newaxis] 
    return np.float32(mfccs)

# ==========================================
# 4. AUDIO ENGINE & REAL-TIME LOOP
# ==========================================
def get_rms(block):
    """Calculates the volume level of an audio chunk."""
    # Convert raw bytes to float array
    audio_data = np.frombuffer(block, dtype=np.int16).astype(np.float32) / 32768.0
    # Calculate Root Mean Square (Volume)
    rms = np.sqrt(np.mean(audio_data**2))
    return rms

def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("\n" + "="*50)
    print(" 🧑‍🦽 RAFEEQ WHEELCHAIR ASSISTANT ONLINE")
    print(" 🎤 Listening in background (Awaiting voice...)")
    print("="*50 + "\n")

    try:
        while True:
            # Read a tiny chunk of audio (about 0.06 seconds)
            data = stream.read(CHUNK, exception_on_overflow=False)
            volume = get_rms(data)
            
            # If the volume spikes above the threshold, someone is speaking!
            if volume > VOLUME_THRESHOLD:
                print(f"🔊 Voice detected! Recording 1.5s...")
                
                # Start recording immediately
                frames = [data] # Keep the first chunk so we don't cut off the first letter
                for _ in range(0, int(SAMPLE_RATE / CHUNK * DURATION)):
                    frames.append(stream.read(CHUNK, exception_on_overflow=False))
                
                print("🧠 Processing...")
                raw_audio = b''.join(frames)
                audio_array = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                
                # 1. Extract Features
                input_data = extract_features(audio_array)
                
                # 2. Run the AI Model (Inference)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                
                # 3. Get the Result
                best_match_index = np.argmax(output_data)
                confidence = output_data[best_match_index]
                command_name = labels[best_match_index]
                
                # Print result
                if confidence > 0.70: # 70% confidence threshold for safety
                    print(f"🟢 COMMAND: {command_name.upper()} (Confidence: {confidence*100:.1f}%)\n")
                else:
                    print(f"🔴 IGNORED: Unsure. (Best guess: {command_name} at {confidence*100:.1f}%)\n")
                    
                print(" 🎤 Listening in background...")
                
    except KeyboardInterrupt:
        print("\nShutting down Rafeeq...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()