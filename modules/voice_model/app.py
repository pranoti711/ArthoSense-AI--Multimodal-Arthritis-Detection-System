import tkinter as tk
from tkinter import filedialog, messagebox
import os

from predict import predict

root = tk.Tk()
root.title("Arthritis Voice Detection System")
root.geometry("500x300")
root.resizable(False, False)

selected_file = tk.StringVar()
result_text = tk.StringVar()
confidence_text = tk.StringVar()


def browse_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("WAV files", "*.wav")]
    )

    if file_path:
        selected_file.set(file_path)
        result_text.set("")
        confidence_text.set("")


def run_prediction():
    file_path = selected_file.get()

    if not file_path:
        messagebox.showerror("Error", "Please select an audio file")
        return

    try:
        label, confidence = predict(file_path)

        result_text.set(f"Prediction: {label}")
        confidence_text.set(f"Confidence: {confidence:.2f}")

    except Exception as e:
        messagebox.showerror("Error", str(e))



title = tk.Label(root, text="🎤 Voice-based Arthritis Detection",
                 font=("Arial", 16, "bold"))
title.pack(pady=10)

file_frame = tk.Frame(root)
file_frame.pack(pady=10)

file_entry = tk.Entry(file_frame, textvariable=selected_file, width=40)
file_entry.pack(side=tk.LEFT, padx=5)

browse_btn = tk.Button(file_frame, text="Browse", command=browse_file)
browse_btn.pack(side=tk.LEFT)


predict_btn = tk.Button(root, text="Analyze Voice",
                        command=run_prediction,
                        bg="blue", fg="white",
                        font=("Arial", 12, "bold"))
predict_btn.pack(pady=15)

# Result display
result_label = tk.Label(root, textvariable=result_text,
                        font=("Arial", 14))
result_label.pack(pady=5)

confidence_label = tk.Label(root, textvariable=confidence_text,
                            font=("Arial", 12))
confidence_label.pack(pady=5)

root.mainloop()


