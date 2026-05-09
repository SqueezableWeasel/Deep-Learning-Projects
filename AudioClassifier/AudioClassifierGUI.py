import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import csv
import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import os

SAMPLE_RATE = 16000
RECORD_SECONDS = 5
AUDIO_FILE = "recorded_audio.wav"
BACKGROUND_IMAGE = "background.png"  # Change this to your image path


class YAMNetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YAMNet Audio Classifier")

        # Fill the screen
        self.root.state("zoomed")  # Windows
        # self.root.attributes("-fullscreen", True)  # macOS/Linux alternative

        self.audio = None
        self.model = None
        self.class_names = []

        # Background canvas
        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.bg_original = Image.open(BACKGROUND_IMAGE)
        self.bg_image = None
        self.bg_item = self.canvas.create_image(0, 0, anchor="nw")

        self.canvas.bind("<Configure>", self.resize_background)

        # Foreground panel
        self.panel = tk.Frame(self.canvas, bg="#f0f0f0", bd=4, relief="ridge")
        self.panel_window = self.canvas.create_window(
            0, 0, window=self.panel, anchor="center"
        )

        self.status = tk.Label(
            self.panel,
            text="Loading YAMNet...",
            font=("Arial", 18),
            bg="#f0f0f0"
        )
        self.status.pack(pady=20)

        self.record_btn = tk.Button(
            self.panel,
            text="Record Audio",
            font=("Arial", 16),
            width=22,
            command=self.record_audio
        )
        self.record_btn.pack(pady=10)

        self.play_btn = tk.Button(
            self.panel,
            text="Play Back",
            font=("Arial", 16),
            width=22,
            command=self.play_audio
        )
        self.play_btn.pack(pady=10)

        self.classify_btn = tk.Button(
            self.panel,
            text="Classify Audio",
            font=("Arial", 16),
            width=22,
            command=self.classify_audio
        )
        self.classify_btn.pack(pady=10)

        self.results = tk.Text(
            self.panel,
            height=12,
            width=60,
            font=("Consolas", 12)
        )
        self.results.pack(padx=25, pady=25)

        threading.Thread(target=self.load_model, daemon=True).start()

    def resize_background(self, event):
        width = event.width
        height = event.height

        resized = self.bg_original.resize((width, height), Image.LANCZOS)
        self.bg_image = ImageTk.PhotoImage(resized)

        self.canvas.itemconfig(self.bg_item, image=self.bg_image)

        self.canvas.coords(self.panel_window, width // 2, height // 2)

    def load_model(self):
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")

        class_map_path = self.model.class_map_path().numpy().decode("utf-8")

        with tf.io.gfile.GFile(class_map_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.class_names.append(row["display_name"])

        self.status.config(text="YAMNet loaded. Ready.")

    def record_audio(self):
        threading.Thread(target=self._record_audio, daemon=True).start()

    def _record_audio(self):
        self.status.config(text=f"Recording for {RECORD_SECONDS} seconds...")
        self.results.delete("1.0", tk.END)

        try:
            audio = sd.rec(
                int(RECORD_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            self.audio = audio.flatten()
            sf.write(AUDIO_FILE, self.audio, SAMPLE_RATE)

            self.status.config(text="Recording complete.")

        except Exception as e:
            messagebox.showerror("Recording Error", str(e))
            self.status.config(text="Recording failed.")

    def play_audio(self):
        if self.audio is None:
            messagebox.showwarning("No Audio", "Please record audio first.")
            return

        threading.Thread(target=self._play_audio, daemon=True).start()

    def _play_audio(self):
        self.status.config(text="Playing audio...")
        sd.play(self.audio, SAMPLE_RATE)
        sd.wait()
        self.status.config(text="Playback complete.")

    def classify_audio(self):
        if self.audio is None:
            messagebox.showwarning("No Audio", "Please record audio first.")
            return

        if self.model is None:
            messagebox.showwarning("Model Not Ready", "YAMNet is still loading.")
            return

        threading.Thread(target=self._classify_audio, daemon=True).start()

    def _classify_audio(self):
        self.status.config(text="Classifying audio...")
        self.results.delete("1.0", tk.END)

        try:
            waveform = self.audio.astype(np.float32)
            scores, embeddings, spectrogram = self.model(waveform)

            mean_scores = tf.reduce_mean(scores, axis=0)

            top_k = 10
            top_indices = tf.argsort(mean_scores, direction="DESCENDING")[:top_k]

            self.results.insert(tk.END, "Top identified classes:\n\n")

            for i in top_indices:
                class_name = self.class_names[int(i)]
                confidence = float(mean_scores[i])
                self.results.insert(
                    tk.END,
                    f"{class_name:35s} {confidence:.3f}\n"
                )

            self.status.config(text="Classification complete.")

        except Exception as e:
            messagebox.showerror("Classification Error", str(e))
            self.status.config(text="Classification failed.")


if __name__ == "__main__":
    root = tk.Tk()
    app = YAMNetGUI(root)

    # Press Esc to exit fullscreen/zoomed window
    root.bind("<Escape>", lambda event: root.destroy())

    root.mainloop()
