That is fantastic news! I am thrilled to hear the local inference script worked well for you. Building a custom AI from scratch and running it with zero latency is a huge milestone for your graduation project.

Since you are uploading this to GitHub, your `README.md` needs to look professional. It should explain *why* you chose this approach (TinyML over generic models like Vosk) and provide clear instructions so anyone (including your professors or future employers) can understand and run your code.

Here is a complete, well-structured `README.md` file ready for you to copy and paste into your GitHub repository.

---

```markdown
# 🧑‍🦽 Rafeeq Smart Wheelchair: Offline Voice Control (TinyML)

This repository contains the custom **Voice Command Recognition System** built for "Rafeeq", a smart wheelchair designed for users with limited mobility. 

Because the system controls a physical vehicle, it prioritizes **safety, ultra-low latency, and 100% offline edge-computing**. To achieve this, we bypassed heavy, generic Speech-to-Text engines (which struggle with Egyptian Arabic dialects and cause lag) and built a **Custom TinyML Keyword Spotting (KWS)** system.



## ✨ Key Features
* **100% Offline (Edge Computing):** Runs entirely without internet using a lightweight `.tflite` model.
* **Ultra-Low Latency:** Uses Voice Activity Detection (VAD) to idle the CPU, activating only when speech is detected to process commands in under ~50 milliseconds.
* **Dialect-Friendly:** Trained specifically on custom Egyptian Arabic commands.
* **High Safety:** Rejects background noise and requires a strict confidence threshold to trigger wheelchair movements.

## 📂 Repository Structure
* `Data_Collection_and_Training.ipynb`: A Google Colab notebook that uses a JavaScript bridge to record audio directly from your browser, saves it to Google Drive, extracts MFCC features, and trains a Convolutional Neural Network (CNN).
* `main.py`: The local PC inference script. It listens to the microphone in the background and runs real-time classification when a voice is detected.
* `rafeeq_model.tflite`: The compressed, edge-ready neural network model.
* `labels.txt`: The mapped vocabulary list (e.g., Move Forward, Stop, Turn Left).

## 🧠 The 10-Word Vocabulary
The model is strictly constrained to the following commands to maximize accuracy:
1. `rafeeq` (Wake Word)
2. `sleep` / `stop`
3. `move_forward`
4. `move_backward`
5. `turn_left`
6. `turn_right`
7. `go_to_bedroom`
8. `go_to_bathroom`
9. `go_to_kitchen`
10. `go_to_livingroom`

---

## 🚀 How to Recreate the Model (Google Colab)
If you want to train this model on your own voice:
1. Open the `Data_Collection_and_Training.ipynb` file in Google Colab.
2. Run the first cell to mount your Google Drive.
3. Use the integrated UI buttons to record 10-15 samples of each command. The script will automatically convert them to `16kHz, 16-bit Mono WAV` format using FFmpeg.
4. Run the training cell. The script will extract **MFCCs (Mel Frequency Cepstral Coefficients)**, train a tiny CNN, and export a `rafeeq_model.tflite` file to your Drive.

---

## 💻 How to Run Locally (PC/Laptop Validation)

### 1. Install Dependencies
Create a virtual environment (e.g., using Conda) and install the required libraries:
```bash
pip install tensorflow librosa pyaudio numpy

```

### 2. File Placement

Ensure your directory looks like this:

```text
Rafeeq_Project/
 ├── main.py
 ├── rafeeq_model.tflite
 └── labels.txt

```

### 3. Run the Inference Engine

Navigate to your folder in the terminal and execute the script:

```bash
python main.py

```

* **How it works:** The script will sit silently (`Listening in background...`). The moment you speak, the Volume Threshold triggers, it records exactly 1.5 seconds of audio, extracts the MFCCs, and passes it through the TFLite model for an instant prediction.

## 🔜 Future Deployment (Raspberry Pi)

This project is engineered to be deployed on a Raspberry Pi. For the hardware deployment phase, the massive `tensorflow` library will be replaced with the ultra-lightweight `tflite-runtime` library to ensure zero RAM overflow and maximum processing speed.

```

***

### Next Steps
Once you have this pushed to GitHub, you officially have a fully functioning PC prototype for the Speech Module! 

According to your schedule, the next major milestone is **Raspberry Pi Migration (Days 5-6)**. Would you like me to guide you through setting up the Raspberry Pi OS and installing `tflite-runtime` so we can move this code onto the actual wheelchair hardware?

```
