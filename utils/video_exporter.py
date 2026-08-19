import imageio
import os
import time
import numpy as np

class VideoExporter:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.recording = False
        self.writer = None
        self.output_path = ""
        
    def start_recording(self, output_filename=None, fps=30):
        if self.recording:
            return
            
        self.recording = True
        if output_filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_filename = f"recording_{timestamp}.mp4"
            
        self.output_path = os.path.join(self.output_dir, output_filename)
        self.writer = imageio.get_writer(self.output_path, fps=fps)
        print(f"[VideoExporter] Started recording to {self.output_path}")

    def add_frame(self, frame_np):
        if self.recording and self.writer is not None:
            # frame_np is expected to be (H, W, 3) float32 in [0, 1]
            if frame_np.dtype == np.float32 or frame_np.dtype == np.float64:
                frame_uint8 = (np.clip(frame_np, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                frame_uint8 = frame_np
            self.writer.append_data(frame_uint8)

    def stop_recording(self):
        if self.recording:
            self.recording = False
            if self.writer is not None:
                self.writer.close()
                self.writer = None
            print(f"[VideoExporter] Stopped recording. Saved to {self.output_path}")
