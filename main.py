#!/usr/bin/env python3
"""
rafeeq_pipeline.py
==================
Production-ready hybrid voice control pipeline for the Rafeeq wheelchair.

Audio pipeline (two-model design):
  Model 1 – KWS / Local model   : rafeeq_model.tflite  (always on, detects
                                    'rafeeq' wake-word and 'stop' safety-stop)
  Model 2 – openai-whisper       : Python whisper library transcribes the 3-s
                                    command window after the wake-word fires.
             text_classifier      : text_classifier.tflite maps transcript
                                    to one of 11 wheelchair commands.

Architecture:
  Layer 1 – KWS Safety Net  : TFLite CNN (rafeeq_model.tflite)
                               detects wake-word "Rafeeq" via MFCC features.
  Layer 2 – Speech-to-Text  : whisper.cpp binary transcribes a 3-second command window.
  Layer 3 – Intent Mapping  : TFLite FC model (text_classifier.tflite) maps
                               Whisper text → 11 wheelchair commands.

Target Hardware : Raspberry Pi 5
Audio Backend   : PyAudio (PortAudio)
Python          : 3.9+
"""

# ──────────────────────────────────────────────────────────────────────────────
# STD LIBRARY
# ──────────────────────────────────────────────────────────────────────────────
import json
import logging
import re
import sys
import tempfile
import wave
from pathlib import Path
from typing import List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY  (tflite-runtime for Pi, numpy, pyaudio, librosa, openai-whisper)
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pyaudio

try:
    # Option 1 – ai_edge_litert  ← Google's official tflite_runtime replacement.
    #   Supports all modern TFLite op versions (FULLY_CONNECTED v12+).
    #   Install: pip install ai-edge-litert
    from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter  # type: ignore
except ImportError:
    try:
        # Option 2 – legacy tflite_runtime (older Pi deployments).
        #   WARNING: versions < 2.14 do NOT support FULLY_CONNECTED op v12,
        #   causing a "Didn't find op" error at model load.  Prefer ai_edge_litert.
        #   Install: pip install tflite-runtime
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    except ImportError:
        try:
            # Option 3 – full TensorFlow bundled interpreter (dev / Windows).
            from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterpreter  # type: ignore
        except ImportError:
            sys.exit(
                "[FATAL] No TFLite interpreter found.\n"
                "Install one of:\n"
                "  pip install ai-edge-litert   # Raspberry Pi 5 (recommended)\n"
                "  pip install tflite-runtime   # older Pi (must be >= 2.14)\n"
                "  pip install tensorflow       # Windows / dev machine\n"
                "\n"
                "Make sure you run this script with the SAME Python that has\n"
                "TensorFlow installed (e.g. your miniconda env, not py3.13).\n"
                "  C:\\Users\\OozZ_\\miniconda3\\python.exe main.py"
            )

try:
    import librosa
except ImportError:
    sys.exit("[FATAL] librosa is not installed. Run: pip install librosa")

try:
    from pywhispercpp.model import Model as _WhisperCppModel
except ImportError:
    sys.exit(
        "[FATAL] pywhispercpp is not installed.\n"
        "  Run: pip install pywhispercpp\n"
        "  Then download a real GGML model (the for-tests-*.bin stubs are NOT real models):\n"
        "    Linux/Pi : bash whisper.cpp/models/download-ggml-model.sh base\n"
        "    Windows  : download ggml-base.bin from huggingface.co/ggerganov/whisper.cpp"
    )

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Rafeeq")

# ──────────────────────────────────────────────────────────────────────────────
# ❶  CONFIGURATION  (edit these to match your deployment)
# ──────────────────────────────────────────────────────────────────────────────

# --- Paths ---
KWS_MODEL_PATH       = Path("rafeeq_model.tflite")      # Layer 1: wake-word CNN  (local model)
INTENT_MODEL_PATH    = Path("text_classifier.tflite")   # Layer 3: intent FC model
METADATA_PATH        = Path("metadata.json")            # vocab list + label list (model source of truth)
TEMP_COMMAND_WAV     = Path(tempfile.gettempdir()) / "rafeeq_command.wav"

# --- Layer 2: whisper.cpp (via pywhispercpp Python bindings) ---
# Path to the local GGML model file inside the cloned whisper.cpp repo.
# WARNING: The for-tests-*.bin files (~575 KB) are stubs — NOT real models.
# Download a real model first:
#   Linux/Pi : bash whisper.cpp/models/download-ggml-model.sh base
#   Windows  : download ggml-base.bin from huggingface.co/ggerganov/whisper.cpp
#              and place it in whisper.cpp/models/
# Use ggml-tiny.bin (~75 MB) for maximum speed on Pi 5,
# or ggml-base.bin (~148 MB) for better Arabic accuracy.
WHISPER_MODEL_PATH   = Path("whisper.cpp/models/ggml-base.bin")
WHISPER_NUM_THREADS  = 4               # CPU threads (4 = Pi 5 core count)

# ── DEBUG FLAG ─────────────────────────────────────────────────────────────────
# Set True to print the raw whisper transcript to the console before it is
# passed to the intent classifier.  Leave True during development / testing.
DEBUG_SHOW_WHISPER_OUTPUT: bool = True

# --- Audio ---
SAMPLE_RATE          = 16_000          # Hz – must match training
CHANNELS             = 1               # Mono
CHUNK_FRAMES         = 512            # PyAudio read chunk (~32 ms @16 kHz)
FORMAT               = pyaudio.paInt16 # 16-bit PCM

# --- Layer 1: KWS ---
KWS_WINDOW_SECS      = 1.5            # seconds of audio analysed per KWS call
KWS_CONFIDENCE_THRESH = 0.80          # minimum probability to accept "Rafeeq"
VOLUME_THRESHOLD     = 300            # RMS amplitude below this → silence (VAD gate)

# MFCC parameters – must match Colab training exactly
N_MFCC               = 13
KWS_INPUT_SHAPE      = (1, N_MFCC, 47, 1)   # [batch, mfcc, time, channel]

# --- Layer 2: Whisper ---
COMMAND_WINDOW_SECS  = 3              # seconds recorded after wake-word fires
WHISPER_LANGUAGE     = "ar"          # Arabic; use "en" for English

# --- Layer 3: Intent ---
INTENT_LABELS = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STOP",
    "GO_TO_KITCHEN",
    "GO_TO_BATHROOM",
    "GO_TO_BEDROOM",
    "GO_TO_LIVING_ROOM",
    "SPEED_UP",
    "SLOW_DOWN",
]
INTENT_CONFIDENCE_THRESH = 0.80       # minimum probability for a valid command (safety requirement)

# ──────────────────────────────────────────────────────────────────────────────
# ❷  UTILITY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def rms_amplitude(audio_bytes: bytes) -> float:
    """Compute root-mean-square amplitude of raw 16-bit PCM bytes."""
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0


def pcm_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit PCM bytes → normalised float32 array in [-1, 1]."""
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    return samples / 32768.0


def save_wav(filepath: Path, audio_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    """Persist raw PCM bytes as a mono 16-bit WAV file."""
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)          # 2 bytes = 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)


def extract_mfcc(audio_float: np.ndarray,
                 target_frames: int = 47) -> np.ndarray:
    """
    Extract MFCC features and reshape to CNN input tensor.

    Steps:
      1. Trim leading/trailing silence.
      2. Pad or truncate to exactly `target_frames` time-steps.
      3. Compute N_MFCC coefficients.
      4. Return tensor shaped [1, N_MFCC, target_frames, 1].
    """
    # 1. Trim silence
    audio_trimmed, _ = librosa.effects.trim(audio_float, top_db=20)

    # 2. Pad / truncate to fixed length
    target_samples = int(KWS_WINDOW_SECS * SAMPLE_RATE)
    if len(audio_trimmed) < target_samples:
        pad_width = target_samples - len(audio_trimmed)
        audio_trimmed = np.pad(audio_trimmed, (0, pad_width), mode="constant")
    else:
        audio_trimmed = audio_trimmed[:target_samples]

    # 3. MFCC extraction
    mfccs = librosa.feature.mfcc(
        y=audio_trimmed,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=512,
        hop_length=int(SAMPLE_RATE * KWS_WINDOW_SECS / target_frames),
    )   # shape: (N_MFCC, actual_time_steps)

    # 4. Pad/trim time axis to exactly target_frames
    if mfccs.shape[1] < target_frames:
        mfccs = np.pad(mfccs, ((0, 0), (0, target_frames - mfccs.shape[1])), mode="constant")
    else:
        mfccs = mfccs[:, :target_frames]

    # 5. Reshape to [1, N_MFCC, target_frames, 1]
    return mfccs.reshape(KWS_INPUT_SHAPE).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# ❸  LAYER 1 – KWS (Wake-Word Detection)
# ──────────────────────────────────────────────────────────────────────────────

class KWSDetector:
    """
    Wraps the TFLite CNN always-on keyword model (rafeeq_model.tflite).

    This model runs continuously in the background at very low CPU cost.
    It serves two purposes:
      • Detects "rafeeq" → activates the Whisper command window (Layer 2+3).
      • Detects "stop"   → triggers an IMMEDIATE safety-stop, bypassing Whisper.

    Class indices are loaded directly from labels.txt so they always stay
    in sync with however the model was trained.
    """

    # Words the KWS layer handles directly (no Whisper needed for these)
    IMMEDIATE_COMMANDS = {"stop", "rafeeq"}

    def __init__(self, model_path: Path = KWS_MODEL_PATH,
                 labels_path: Path = Path("labels.txt")):
        if not model_path.exists():
            raise FileNotFoundError(f"KWS model not found at: {model_path}")

        # Load labels so indices stay in sync with the trained model
        if labels_path.exists():
            raw = labels_path.read_text(encoding="utf-8").splitlines()
            self._labels = [l.strip().lower() for l in raw if l.strip()]
        else:
            # Fallback: match labels.txt order we know from training
            self._labels = [
                "go_to_bathroom", "go_to_bedroom", "go_to_kitchen",
                "go_to_livingroom", "move_backward", "move_forward",
                "rafeeq", "sleep", "stop", "turn_left", "turn_right",
            ]
            log.warning("labels.txt not found – using hardcoded label order.")

        self._interp = TFLiteInterpreter(model_path=str(model_path))
        self._interp.allocate_tensors()
        self._in_idx  = self._interp.get_input_details()[0]["index"]
        self._out_idx = self._interp.get_output_details()[0]["index"]

        # Locate the class indices for the two critical keywords
        self._rafeeq_idx = self._labels.index("rafeeq") if "rafeeq" in self._labels else -1
        self._stop_idx   = self._labels.index("stop")   if "stop"   in self._labels else -1
        log.info("KWS model loaded | labels=%s | rafeeq_idx=%d | stop_idx=%d",
                 self._labels, self._rafeeq_idx, self._stop_idx)

    def predict(self, mfcc_tensor: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Run inference on a pre-processed MFCC tensor.

        Returns:
            (detected_keyword (str | None), confidence (float))

            detected_keyword is:
              'rafeeq' → open the Whisper command window
              'stop'   → immediate safety stop, skip Whisper
              None     → background noise, keep listening
        """
        self._interp.set_tensor(self._in_idx, mfcc_tensor)
        self._interp.invoke()
        probs = self._interp.get_tensor(self._out_idx)[0]   # (num_classes,)

        best_idx   = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        if confidence < KWS_CONFIDENCE_THRESH:
            return None, confidence

        label = self._labels[best_idx] if best_idx < len(self._labels) else "unknown"

        # Only act on the two keywords the KWS layer owns
        if label in self.IMMEDIATE_COMMANDS:
            return label, confidence

        return None, confidence


# ──────────────────────────────────────────────────────────────────────────────
# ❹  LAYER 2 – WHISPER SPEECH-TO-TEXT  (openai-whisper Python library)
# ──────────────────────────────────────────────────────────────────────────────

# Lazy-loaded whisper.cpp model instance (pywhispercpp)
_whisper_cpp_model: Optional[object] = None

def _get_whisper_model():
    """Lazy-load the pywhispercpp model from the local GGML file."""
    global _whisper_cpp_model
    if _whisper_cpp_model is None:
        if not WHISPER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"[Whisper.cpp] Model not found: {WHISPER_MODEL_PATH}\n"
                "Download the real model first:\n"
                "  Linux/Pi : bash whisper.cpp/models/download-ggml-model.sh base\n"
                "  Windows  : download ggml-base.bin from huggingface.co/ggerganov/whisper.cpp"
            )
        log.info("[Whisper.cpp] Loading model from %s …", WHISPER_MODEL_PATH)
        _whisper_cpp_model = _WhisperCppModel(
            str(WHISPER_MODEL_PATH),
            language=WHISPER_LANGUAGE,
            n_threads=WHISPER_NUM_THREADS,
            print_realtime=False,
            print_progress=False,
        )
        log.info("[Whisper.cpp] Model ready.")
    return _whisper_cpp_model


def transcribe_with_whisper(audio_bytes: bytes) -> Optional[str]:
    """
    Transcribe raw PCM audio using whisper.cpp via the pywhispercpp bindings.

    Accepts raw 16-bit PCM bytes directly (no ffmpeg, no temp file needed).
    The bytes are converted to a 1-D float32 numpy array and passed straight
    to pywhispercpp's transcribe() which accepts numpy arrays natively.

    Flow:
      Layer 1 (KWS) detected 'rafeeq' → 3 s of audio recorded → here.
      Whisper.cpp transcript is printed to console (DEBUG_SHOW_WHISPER_OUTPUT)
      BEFORE being forwarded to the intent classifier (Layer 3).

    Returns None on failure so the pipeline can gracefully recover.
    """
    try:
        model = _get_whisper_model()

        # Convert raw 16-bit PCM bytes → 1-D float32 in [-1, 1]
        audio_float32 = (
            np.frombuffer(audio_bytes, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )

        # pywhispercpp transcribe() accepts a 1-D float32 numpy array directly;
        # returns a list of Segment objects each having a .text attribute.
        segments = model.transcribe(audio_float32)
        raw_text = " ".join(seg.text for seg in segments).strip()

        # ── DEBUG: show raw whisper.cpp output BEFORE intent classifier ───────
        if DEBUG_SHOW_WHISPER_OUTPUT:
            print("\n" + "═" * 60)
            print("  [WHISPER.CPP RAW OUTPUT]  (before intent classifier)")
            print("═" * 60)
            print(f"  Text : {raw_text!r}")
            print("═" * 60 + "\n")

        # Strip common noise markers whisper.cpp sometimes emits
        for noise in ("[BLANK_AUDIO]", "(music)", "(noise)", "[MUSIC]", "[NOISE]",
                      "[ Silence ]", "[silence]"):
            raw_text = raw_text.replace(noise, "")
        raw_text = raw_text.strip()

        if raw_text:
            log.info("[Whisper.cpp] Transcript → '%s'", raw_text)
        else:
            log.warning("[Whisper.cpp] Empty transcript (silence / noise).")

        return raw_text if raw_text else None

    except Exception as exc:
        log.error("[Whisper.cpp] Transcription error: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# ❺  LAYER 3 – INTENT CLASSIFICATION (Bag-of-Words + FC TFLite)
# ──────────────────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Converts Whisper transcript → normalise → Bag-of-Words vector → TFLite FC inference.

    Loads vocab and label order from metadata.json (the single source of truth
    generated by text_model_creator.ipynb) so vocab indices and class indices
    always stay in sync with the trained model.

    metadata.json format:
        { "vocab": ["word1", "word2", ...], "labels": ["intent1", ...] }
        vocab:  list index = BoW vector index
        labels: list index = model output class index
    """

    def __init__(self,
                 model_path: Path = INTENT_MODEL_PATH,
                 metadata_path: Path = METADATA_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Intent model not found at: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

        # Load vocabulary and label order — the single source of truth
        with metadata_path.open("r", encoding="utf-8") as f:
            meta: dict = json.load(f)

        vocab_list: list   = meta["vocab"]    # list → position = BoW index
        self._labels: list = meta["labels"]   # list → position = model class index

        # Build word → index dict for O(1) lookup at inference time
        self._vocab: dict = {word: idx for idx, word in enumerate(vocab_list)}

        # Load TFLite interpreter
        self._interp = TFLiteInterpreter(model_path=str(model_path))
        self._interp.allocate_tensors()
        self._in_idx  = self._interp.get_input_details()[0]["index"]
        self._out_idx = self._interp.get_output_details()[0]["index"]

        # Model's input dimension is the ground truth for BoW vector size
        self._bow_size = int(self._interp.get_input_details()[0]["shape"][1])

        if len(vocab_list) != self._bow_size:
            log.warning(
                "metadata.json has %d vocab tokens but model input expects %d. "
                "Tokens with index >= %d will be silently ignored.",
                len(vocab_list), self._bow_size, self._bow_size,
            )

        log.info("Intent model loaded | dim: %d | vocab: %d tokens | labels: %s",
                 self._bow_size, len(vocab_list), self._labels)

    @staticmethod
    def _normalize_arabic(text: str) -> str:
        """
        Normalise Egyptian Arabic text before BoW vectorisation.

        Applies the same 6-step pipeline used during training (text_model_creator.ipynb)
        so formal and colloquial variants of the same word hit the same vocab token.

          1. Remove diacritics (harakat/tashkeel U+064B–U+0652) and tatweel (U+0640).
          2. Unify Alef variants  (أ إ آ ٱ → ا).
          3. Unify Ya tail        (ى → ي).
          4. Unify Hamza seats    (ؤ → و,  ئ → ي).
          5. Strip punctuation/numbers (keep Arabic block + Latin + whitespace).
          6. Strip common prefixes (بال لل ال في و) — longest first,
             one prefix per word, minimum 2 chars must remain after strip.
        """
        # 1. Diacritics & tatweel
        text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
        # 2. Alef normalisation
        text = re.sub(r'[أإآٱ]', 'ا', text)
        # 3. Ya normalisation
        text = text.replace('ى', 'ي')
        # 4. Hamza seats
        text = text.replace('ؤ', 'و').replace('ئ', 'ي')
        # 5. Strip non-Arabic / non-Latin punctuation
        text = re.sub(r'[^\u0600-\u06FFa-zA-Z\s]', '', text)
        # 6. Prefix stripping — longest first so 'بال' beats 'ال'
        prefixes = ['بال', 'لل', 'ال', 'في', 'و']
        cleaned = []
        for word in text.split():
            for prefix in prefixes:
                if word.startswith(prefix) and len(word) >= len(prefix) + 2:
                    word = word[len(prefix):]
                    break  # one prefix per word
            cleaned.append(word)
        return ' '.join(cleaned).strip()

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        """Standard Levenshtein edit distance, O(min(m,n)) space."""
        if len(a) < len(b):
            a, b = b, a
        if not b:
            return len(a)
        row = list(range(len(b) + 1))
        for ca in a:
            prev, row[0] = row[0], row[0] + 1
            for j, cb in enumerate(b, 1):
                prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
        return row[len(b)]

    def _fuzzy_vocab_lookup(self, token: str) -> Optional[int]:
        """
        Fuzzy fallback for tokens that were not found by exact vocab lookup.

        Handles ASR mis-transcriptions such as:
          'حنان' → 'حمام'  (edit distance 2 — two ن→م substitutions)

        Conservative thresholds (safety-critical wheelchair context):
          word length == 3  →  max edit distance 1
          word length >= 4  →  max edit distance 2
          word length  < 3  →  no fuzzy matching (too ambiguous)

        Exact vocab matches always take priority; this method is only
        called when the exact lookup returns None.  A WARNING is logged
        every time a fuzzy match fires so false positives surface in testing.
        """
        tlen = len(token)
        if tlen < 3:
            return None
        max_dist = 1 if tlen == 3 else 2

        best_dist = max_dist + 1   # sentinel — means "no match yet"
        best_idx: Optional[int] = None
        best_word: Optional[str] = None

        for word, idx in self._vocab.items():
            # Fast length pre-filter: skip words that cannot possibly be within
            # max_dist edits (length difference alone exceeds the budget).
            if abs(len(word) - tlen) > max_dist:
                continue
            d = self._levenshtein(token, word)
            if d < best_dist:
                best_dist, best_idx, best_word = d, idx, word

        if best_idx is not None:
            log.warning(
                "[Intent] Fuzzy match: '%s' → '%s' (edit_dist=%d) — verify this is correct!",
                token, best_word, best_dist,
            )
        return best_idx

    def _text_to_bow(self, text: str) -> np.ndarray:
        """Normalise `text` then encode as a float32 Bag-of-Words vector.

        For each token: try exact vocab lookup first; if not found, fall back
        to Levenshtein fuzzy matching so ASR variants like 'حنان' still map
        to the correct vocab entry ('حمام').
        """
        bow = np.zeros((1, self._bow_size), dtype=np.float32)
        for token in self._normalize_arabic(text).split():
            idx = self._vocab.get(token)          # exact match (preferred)
            if idx is None:
                idx = self._fuzzy_vocab_lookup(token)   # fuzzy fallback
            if idx is not None and idx < self._bow_size:
                bow[0, idx] += 1.0
        return bow

    def classify(self, text: str) -> Tuple[Optional[str], float]:
        """
        Normalise → BoW → TFLite inference → return (intent_label, confidence).

        Returns (None, confidence) when confidence < INTENT_CONFIDENCE_THRESH
        so the pipeline can safely discard ambiguous commands.
        """
        bow_vector = self._text_to_bow(text)
        self._interp.set_tensor(self._in_idx, bow_vector)
        self._interp.invoke()
        probs = self._interp.get_tensor(self._out_idx)[0]   # shape: (num_classes,)

        best_idx   = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        if confidence < INTENT_CONFIDENCE_THRESH:
            log.warning(
                "[Intent] Confidence %.1f%% < %.0f%% threshold – returning UNKNOWN.",
                confidence * 100, INTENT_CONFIDENCE_THRESH * 100,
            )
            return None, confidence

        # Map class index → uppercase label (e.g. "move_forward" → "MOVE_FORWARD")
        # which matches the keys expected by dispatch_command().
        label = self._labels[best_idx].upper() if best_idx < len(self._labels) else "UNKNOWN"
        return label, confidence


# ──────────────────────────────────────────────────────────────────────────────
# ❻  AUDIO RECORDER
# ──────────────────────────────────────────────────────────────────────────────

class AudioRecorder:
    """Manages a persistent PyAudio stream for continuous background listening."""

    def __init__(self):
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_FRAMES,
        )
        log.info("🎙  Audio stream opened @ %d Hz", SAMPLE_RATE)

    def read_chunk(self) -> bytes:
        """Read one CHUNK_FRAMES-sized audio chunk from the microphone."""
        return self._stream.read(CHUNK_FRAMES, exception_on_overflow=False)

    def record_seconds(self, duration_secs: float) -> bytes:
        """
        Block and record exactly `duration_secs` of audio.

        Returns raw PCM bytes.
        """
        num_chunks = int((SAMPLE_RATE / CHUNK_FRAMES) * duration_secs)
        frames: List[bytes] = []
        for _ in range(num_chunks):
            frames.append(self._stream.read(CHUNK_FRAMES, exception_on_overflow=False))
        return b"".join(frames)

    def close(self):
        """Release the PyAudio stream and terminate the host API."""
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._pa.terminate()
        log.info("🎙  Audio stream closed.")


# ──────────────────────────────────────────────────────────────────────────────
# ❼  MAIN PIPELINE LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """
    Infinite background-listening loop.

    ┌─────────────────────────────────────────────────────────────────┐
    │  TWO-MODEL PIPELINE                                             │
    │                                                                 │
    │  MODEL 1 – Local KWS (rafeeq_model.tflite)                     │
    │    • Runs continuously, very low CPU cost.                      │
    │    • Detects 'stop'   → immediate safety stop (no Whisper).     │
    │    • Detects 'rafeeq' → wake-word confirmed, hand off to Whisper│
    │    • Anything else    → back to IDLE.                           │
    │                                                                 │
    │  MODEL 2 – whisper.cpp  +  text_classifier.tflite              │
    │    • Activated ONLY after 'rafeeq' is detected.                 │
    │    • Records 3 s of command audio.                              │
    │    • whisper.cpp → raw Arabic transcript (printed to console).  │
    │    • text_classifier → intent label → dispatch_command().       │
    └─────────────────────────────────────────────────────────────────┘
    """
    # ── Initialise all layers ────────────────────────────────────────────────
    try:
        kws      = KWSDetector()         # Layer 1 – always-on local model
        intent   = IntentClassifier()    # Layer 3 – intent FC model
        recorder = AudioRecorder()
    except FileNotFoundError as exc:
        log.critical("Missing resource: %s", exc)
        sys.exit(1)

    # Pre-load the whisper.cpp model now so the first wake-word isn't slow.
    log.info("[Whisper.cpp] Pre-loading model from %s …", WHISPER_MODEL_PATH)
    _get_whisper_model()

    log.info("═" * 60)
    log.info("  Rafeeq Voice Control – Listening …")
    log.info("  MODEL 1 (KWS)         : always on – detects 'rafeeq' / 'stop'")
    log.info("  MODEL 2 (Whisper.cpp) : activates on wake-word → %s", WHISPER_MODEL_PATH)
    if DEBUG_SHOW_WHISPER_OUTPUT:
        log.info("  DEBUG : Whisper raw output will be printed to console")
    log.info("═" * 60)

    # Pre-allocate KWS capture buffer
    kws_total_chunks  = int((SAMPLE_RATE / CHUNK_FRAMES) * KWS_WINDOW_SECS)
    kws_buffer: List[bytes] = []

    try:
        while True:
            # ── ❶ IDLE: VAD gate ─────────────────────────────────────────────
            chunk = recorder.read_chunk()
            if rms_amplitude(chunk) < VOLUME_THRESHOLD:
                # Silence – discard and loop back (saves CPU)
                continue

            # ── ❶→❷ Volume spike: fill KWS window ────────────────────────────
            log.debug("Volume spike detected – filling KWS window …")
            kws_buffer.clear()
            kws_buffer.append(chunk)
            for _ in range(kws_total_chunks - 1):
                kws_buffer.append(recorder.read_chunk())

            raw_kws_audio = b"".join(kws_buffer)
            audio_float   = pcm_bytes_to_float32(raw_kws_audio)
            mfcc_tensor   = extract_mfcc(audio_float)

            # ── ❷ LAYER 1: KWS INFERENCE (local model) ───────────────────────
            kws_keyword, kws_conf = kws.predict(mfcc_tensor)
            log.info(
                "[KWS] keyword=%-10s | confidence=%.1f%%",
                kws_keyword or "none",
                kws_conf * 100,
            )

            # ── ❸a 'stop' → IMMEDIATE STOP (KWS handles this, no Whisper) ─────
            if kws_keyword == "stop":
                log.info("[KWS] ⛔ STOP detected – immediate safety stop!")
                dispatch_command("STOP")
                continue   # back to idle

            # ── ❸b 'rafeeq' → WAKE-WORD confirmed, hand off to Whisper ────────
            if kws_keyword != "rafeeq":
                continue   # not a keyword we handle – back to idle

            # ─────────────────────────────────────────────────────────────────
            #  FROM HERE ON: MODEL 2 pipeline  (Whisper → Intent Classifier)
            # ─────────────────────────────────────────────────────────────────

            # ── ❹ Record command audio window ─────────────────────────────────
            log.info("[KWS] ✅ Wake-word 'Rafeeq' confirmed – recording %ds …",
                     COMMAND_WINDOW_SECS)
            command_audio = recorder.record_seconds(COMMAND_WINDOW_SECS)
            # No temp file / ffmpeg needed – bytes passed directly to whisper

            # ── ❺ LAYER 2: Whisper transcription ──────────────────────────────
            # Raw output is printed to console (DEBUG_SHOW_WHISPER_OUTPUT=True)
            # BEFORE it is forwarded to the intent classifier.
            transcript = transcribe_with_whisper(command_audio)
            if not transcript:
                log.warning("[Whisper] No usable transcript – back to idle.")
                continue

            # ── ❻ LAYER 3: text_classifier.tflite intent classification ────────
            log.info("[Intent] Classifying transcript → '%s'", transcript)
            command_label, intent_conf = intent.classify(transcript)
            if command_label is None:
                log.warning("[Intent] Confidence too low – command ignored.")
                continue

            # ── ❼ DISPATCH ────────────────────────────────────────────────────
            log.info(
                "[DISPATCH] ✅ %s  (intent confidence: %.1f%%)",
                command_label,
                intent_conf * 100,
            )
            dispatch_command(command_label)

    except KeyboardInterrupt:
        log.info("Shutdown requested by user (Ctrl-C).")
    finally:
        recorder.close()
        if TEMP_COMMAND_WAV.exists():
            TEMP_COMMAND_WAV.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# ❽  COMMAND DISPATCHER
# ──────────────────────────────────────────────────────────────────────────────

def dispatch_command(command: str) -> None:
    """
    Execute the recognised command.

    Replace the stub bodies below with actual GPIO / serial / ROS calls
    to control the wheelchair's motor controller.
    """
    handlers = {
        "MOVE_FORWARD"    : _cmd_move_forward,
        "MOVE_BACKWARD"   : _cmd_move_backward,
        "TURN_LEFT"       : _cmd_turn_left,
        "TURN_RIGHT"      : _cmd_turn_right,
        "STOP"            : _cmd_stop,
        "GO_TO_KITCHEN"   : lambda: _cmd_navigate("kitchen"),
        "GO_TO_BATHROOM"  : lambda: _cmd_navigate("bathroom"),
        "GO_TO_BEDROOM"   : lambda: _cmd_navigate("bedroom"),
        "GO_TO_LIVING_ROOM": lambda: _cmd_navigate("living_room"),
        "SPEED_UP"        : _cmd_speed_up,
        "SLOW_DOWN"       : _cmd_slow_down,
    }
    handler = handlers.get(command)
    if handler:
        handler()
    else:
        log.error("Unknown command '%s' – no handler registered.", command)


# ── Stub command implementations ──────────────────────────────────────────────
# TODO: Replace each stub with real GPIO / serial / ROS 2 calls.

def _cmd_move_forward():
    log.info("[HW] → MOVE FORWARD")

def _cmd_move_backward():
    log.info("[HW] → MOVE BACKWARD")

def _cmd_turn_left():
    log.info("[HW] → TURN LEFT")

def _cmd_turn_right():
    log.info("[HW] → TURN RIGHT")

def _cmd_stop():
    log.info("[HW] → STOP")

def _cmd_navigate(destination: str):
    log.info("[HW] → NAVIGATE TO %s", destination.upper())

def _cmd_speed_up():
    log.info("[HW] → SPEED UP")

def _cmd_slow_down():
    log.info("[HW] → SLOW DOWN")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
