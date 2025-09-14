# (Optional) Tkinter UI front-end
import tkinter as tk
from threading import Thread

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Distracted Driver Detection")
        self.root.geometry("400x250")
        self.current_thread = None
        self.create_widgets()

    def create_widgets(self):
        tk.Button(self.root, text="Eye Detection", command=self.run_eye).pack(pady=10)
        tk.Button(self.root, text="Hand Detection", command=self.run_hand).pack(pady=10)
        tk.Button(self.root, text="Lip Detection", command=self.run_lip).pack(pady=10)
        tk.Button(self.root, text="Phone Detection", command=self.run_phone).pack(pady=10)
        tk.Button(self.root, text="Quit", command=self.quit_app).pack(pady=10)

    def run_eye(self):
        self._run_thread('eye')

    def run_hand(self):
        self._run_thread('hand')

    def run_lip(self):
        self._run_thread('lips')

    def run_phone(self):
        self._run_thread('phone')

    def _run_thread(self, mode):
        if self.current_thread and self.current_thread.is_alive():
            return  # Avoid multiple simultaneous executions
        from drowsiness_detection.interface import runner
        import subprocess
        import sys

        def target():
            subprocess.call([sys.executable, 'drowsiness_detection/interface/runner.py', '--mode', mode])

        self.current_thread = Thread(target=target)
        self.current_thread.start()

    def quit_app(self):
        self.root.quit()

