from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import os
import librosa
import soundfile as sf
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def apply_filter(signal, filter_type, sr):
    nyquist = sr / 2
    if filter_type == "low":
        b, a = scipy.signal.butter(6, 0.1, btype='low')
    elif filter_type == "high":
        b, a = scipy.signal.butter(6, 0.1, btype='high')
    elif filter_type == "band":
        low = 300 / nyquist
        high = 3400 / nyquist
        b, a = scipy.signal.butter(6, [low, high], btype='band')
    else:
        return signal
    return scipy.signal.filtfilt(b, a, signal)

@app.route("/", methods=["GET", "POST"])
def index():
    audio_file = None
    waveform = None
    uploaded_file = None
    selected_filter = None

    if request.method == "POST":
        selected_filter = request.form.get("filter")

        file = request.files.get("audio_file")
        if file and file.filename.endswith(".wav"):
            filename = "uploaded.wav"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            uploaded_file = file.filename

            # Load and process
            signal, sr = librosa.load(filepath, sr=16000)
            filtered_audio = apply_filter(signal, selected_filter, sr)

            # Save enhanced audio
            output_filename = "enhanced.wav"
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
            sf.write(output_path, filtered_audio, sr)
            audio_file = output_filename

            # Generate and save waveform plot
            waveform_path = os.path.join(app.config["UPLOAD_FOLDER"], "waveform.png")
            plt.figure(figsize=(10, 4))
            plt.plot(filtered_audio)
            plt.title("Filtered Audio Waveform")
            plt.tight_layout()
            plt.savefig(waveform_path)
            plt.close()
            waveform = "waveform.png"

    return render_template("index.html",
                           audio_file=audio_file,
                           waveform=waveform,
                           uploaded_file=uploaded_file,
                           selected_filter=selected_filter)

@app.route("/static/<path:filename>")
def download_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
