# Rafeeq (SmartWheel) — Speech Recognition Technical Documentation

> During the development of the Rafeeq smart wheelchair's voice control system, three distinct offline architectures were built, deployed, and benchmarked to determine the most effective solution for real-time Egyptian Arabic command recognition. This process evaluated trade-offs between transcription accuracy, hardware resource consumption, and safety-critical latency.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture A — Local KWS (Selected)](#2-architecture-a--local-kws-selected)
   - [Pipeline Diagram](#pipeline-diagram)
   - [Stage Breakdown](#stage-breakdown)
3. [Architecture B — Vosk (Kaldi)](#3-architecture-b--vosk-kaldi)
4. [Architecture C — Hybrid Pipeline](#4-architecture-c--hybrid-pipeline)
5. [Architecture Comparison](#5-architecture-comparison)
6. [Selection Rationale](#6-selection-rationale)

---

## 1. System Overview

The Rafeeq speech recognition subsystem converts spoken **Egyptian Arabic** into **11 discrete wheelchair control intents**.

| Property | Detail |
| :--- | :--- |
| **Hardware** | Raspberry Pi 5 |
| **Language** | Egyptian Arabic (colloquial) |
| **Commands** | 11 discrete motion intents |
| **Constraint** | Safety-critical latency; shared CPU with navigation & health monitoring |
| **Deployment** | Fully offline (no cloud dependency) |

---

## 2. Architecture A — Local KWS (Selected)

A keyword spotting pipeline custom-trained on Egyptian Arabic wheelchair commands, optimized for low latency and minimal resource usage on embedded hardware.

### Pipeline Diagram

```
┌─────────┐    ┌─────┐    ┌──────┐    ┌─────────┐    ┌──────┐
│   Mic   │───▶│ VAD │───▶│ MFCC │───▶│ CNN     │───▶│ Gate │
│  Input  │    │     │    │      │    │ (TFLite)│    │ ≥80% │
└─────────┘    └──┬──┘    └──────┘    └─────────┘    └──┬───┘
                  │         [1,13,47,1]  ~100 KB          
               discard                              
               if silent              

### Stage Breakdown

#### 2.1 Voice Activity Detection (VAD)

Prevents the neural network from running when no speech is present, keeping idle CPU usage near zero.

- **Method**: Calculates the **Root Mean Square (RMS)** amplitude of the incoming audio buffer.
- **Logic**: Audio frames with RMS below the threshold (`default: 50`) are silently discarded.

#### 2.2 MFCC Feature Extraction

Converts raw audio into a compact spectral representation the CNN can process.

- **What**: **Mel-Frequency Cepstral Coefficients (MFCCs)** — features representing the short-term power spectrum of sound on a perceptual scale.
- **Why the Mel scale**: Mimics human auditory sensitivity by prioritizing lower frequencies where pitch changes are most perceptible.

**Output tensor shape: `[1, 13, 47, 1]`**

| Dimension | Size | Meaning |
| :--- | :---: | :--- |
| Batch | 1 | Single utterance |
| Height | 13 | MFCC coefficients — encodes vocal tract shape |
| Width | 47 | Time frames — horizontal timeline of a 1.5 s window |
| Depth | 1 | Single grayscale channel (reduces memory footprint) |

#### 2.3 TFLite CNN Inference

Classifies the MFCC "image" into one of the 11 command classes.

- **Model type**: Convolutional Neural Network (CNN) — uses spatial filters to detect patterns in the spectrogram.
- **Format**: Quantized **TensorFlow Lite binary** (~100 KB), optimized for edge hardware.
- **Latency**: < 10 ms on Raspberry Pi 5.

#### 2.4 Safety Gate & Dispatch

A final confidence filter before any motor command is issued.

- **Confidence gate**: Command accepted only if model confidence is **≥ 80%**, blocking false triggers from background noise.
- **Dispatch targets**:


---

## 3. Architecture B — Vosk (Kaldi)

An offline Speech-to-Text system using a TDNN acoustic model to decode continuous audio into text, followed by intent matching.

### Pipeline Diagram

```
┌───────┐    ┌──────────────────┐    ┌─────┐    ┌────────┐
│ Audio │───▶│  TDNN Acoustic   │───▶│ STT │───▶│ Intent │
│       │    │      Model       │    │     │    │ Match  │
└───────┘    └──────────────────┘    └─────┘    └────────┘
                  (Kaldi-based)        text
```

| Property | Detail |
| :--- | :--- |
| **Engine** | Vosk (Kaldi backend) |
| **Approach** | Full offline Speech-to-Text → keyword/intent matching |
| **Model size** | ~300 MB |
| **Strength** | Handles continuous speech; language-model guided decoding |
| **Weakness** | High RAM usage; higher latency for short commands |

---

## 4. Architecture C — Hybrid Pipeline

A multi-layer pipeline combining a wake-word detector, Whisper.cpp transcription, and a text classifier with Arabic normalization.

### Pipeline Diagram

```
┌───────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Wake-word │───▶│ Whisper.cpp │───▶│ 6-step Arabic│───▶│ BoW Intent  │
│ Detector  │    │ Transcribe  │    │ Normalization│    │ Classifier  │
└───────────┘    └─────────────┘    └──────────────┘    └─────────────┘
```

**Arabic Normalization Steps (6-stage):**

1. Unicode normalization
2. Diacritic (tashkeel) removal
3. Alef/Ya variant unification
4. Elongation (tatweel) stripping
5. Punctuation removal
6. Colloquial-to-MSA mapping

| Property | Detail |
| :--- | :--- |
| **Engine** | Whisper.cpp + custom BoW classifier |
| **Approach** | Wake-word gated transcription with normalization |
| **Model size** | ~210 MB |
| **Strength** | Robust to colloquial spelling/pronunciation variants |
| **Weakness** | Multi-stage latency; Whisper.cpp transcription adds delay |

---

## 5. Architecture Comparison

| Criterion | A: Local KWS (Selected) | B: Vosk (Kaldi) | C: Hybrid Pipeline |
| :--- | :---: | :---: | :---: |
| **Model size** | ~100 KB | ~300 MB | ~210 MB |
| **RAM usage** | Very low | High | Medium |
| **Inference latency** | < 10 ms | High | Medium–High |
| **Offline capable** | Yes | Yes | Yes |
| **Handles continuous speech** | No (KWS only) | Yes | Yes |
| **Egyptian Arabic support** | Custom-trained | Limited | Normalization-based |
| **Complexity** | Low | Medium | High |
| **Safety-critical suitability** | **Excellent** | Poor | Moderate |

---

## 6. Selection Rationale

**Architecture A (Local TFLite KWS)** was selected as the final architecture based on three criteria:

| Factor | Reason |
| :--- | :--- |
| **Lightweight footprint** | ~100 KB model vs. ~300 MB (B) and ~210 MB (C) — fits entirely in L-cache; no resource contention with navigation or health stacks |
| **Lowest latency** | Sub-10 ms inference enables reliable **STOP** command response — non-negotiable for a mobility device |
| **Custom-trained accuracy** | Trained directly on Egyptian Arabic wheelchair vocabulary; superior precision over general-purpose STT engines for this narrow command set |
