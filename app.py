import cv2
import numpy as np
import csv
import time
import urllib.request
import os
from scipy.signal import butter, filtfilt

# Ensure Haar Cascade XML file exists locally
xml_filename = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_filename):
    print("Downloading face detector XML...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, xml_filename)

face_cascade = cv2.CascadeClassifier(xml_filename)

class StableBioSenseEngine:
    def __init__(self, fps=30, window_size=180):
        self.fps = fps
        self.window_size = window_size  # 6-second sliding buffer
        self.rgb_buffer = []
        self.bpm_history = []
        self.prev_face_box = None

    def smooth_face_box(self, new_box, alpha=0.90):
        """Locks tracking box to prevent movement jitter."""
        if self.prev_face_box is None:
            self.prev_face_box = new_box
            return new_box
        
        x = int(alpha * self.prev_face_box[0] + (1 - alpha) * new_box[0])
        y = int(alpha * self.prev_face_box[1] + (1 - alpha) * new_box[1])
        w = int(alpha * self.prev_face_box[2] + (1 - alpha) * new_box[2])
        h = int(alpha * self.prev_face_box[3] + (1 - alpha) * new_box[3])
        
        self.prev_face_box = (x, y, w, h)
        return self.prev_face_box

    def extract_forehead_skin(self, frame, face_box):
        """Extracts average RGB values from forehead skin patch."""
        x, y, w, h = self.smooth_face_box(face_box)
        fh_y1 = int(y + h * 0.18)
        fh_y2 = int(y + h * 0.35)
        fh_x1 = int(x + w * 0.35)
        fh_x2 = int(x + w * 0.65)

        forehead_roi = frame[fh_y1:fh_y2, fh_x1:fh_x2]
        if forehead_roi.size == 0:
            return None

        cv2.rectangle(frame, (fh_x1, fh_y1), (fh_x2, fh_y2), (0, 255, 0), 2)
        mean_val = cv2.mean(forehead_roi)[:3]
        return mean_val  # (B, G, R)

    def pos_algorithm(self, rgb_signals):
        """Plane-Orthogonal-to-Skin (POS) rPPG extraction."""
        RGB = np.array(rgb_signals)
        mean_rgb = np.mean(RGB, axis=0) + 1e-6
        norm_rgb = RGB / mean_rgb

        S1 = norm_rgb[:, 1] - norm_rgb[:, 2]
        S2 = norm_rgb[:, 1] + norm_rgb[:, 2] - 2 * norm_rgb[:, 0]
        
        h = S1 + (np.std(S1) / (np.std(S2) + 1e-6)) * S2
        return h

    def bandpass_filter(self, signal):
        """Butterworth bandpass filter for human pulse (50 BPM to 160 BPM)."""
        nyq = 0.5 * self.fps
        low = 0.83 / nyq   # 50 BPM
        high = 2.66 / nyq  # 160 BPM
        b, a = butter(2, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def calculate_bpm(self):
        if len(self.rgb_buffer) < self.window_size:
            return None

        bvp = self.pos_algorithm(self.rgb_buffer[-self.window_size:])
        filtered_bvp = self.bandpass_filter(bvp)

        if np.std(filtered_bvp) > 0.4 or np.std(filtered_bvp) < 0.001:
            return self.get_median_bpm(None)

        fft_data = np.abs(np.fft.rfft(filtered_bvp))
        freqs = np.fft.rfftfreq(len(filtered_bvp), 1.0 / self.fps)
        
        valid_idx = np.where((freqs >= 0.83) & (freqs <= 2.66))[0]
        if len(valid_idx) == 0:
            return self.get_median_bpm(None)

        raw_bpm = freqs[valid_idx[np.argmax(fft_data[valid_idx])]] * 60.0
        return self.get_median_bpm(raw_bpm)

    def get_median_bpm(self, new_bpm):
        """Rolling median filter to prevent high/low jumping."""
        if new_bpm is not None and 50 <= new_bpm <= 160:
            self.bpm_history.append(new_bpm)
            if len(self.bpm_history) > 11:
                self.bpm_history.pop(0)

        if len(self.bpm_history) == 0:
            return None

        return float(np.median(self.bpm_history))


# Start Webcam
cap = cv2.VideoCapture(0)
engine = StableBioSenseEngine(fps=30, window_size=180)

# CSV Data Logger
csv_file = open("pulse_telemetry_log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp_s", "Heart_Rate_BPM"])

print("--- Starting BioSense Tracker Pro ---")
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    current_time = time.time() - start_time

    if len(faces) > 0:
        face_box = faces[0]
        mean_bgr = engine.extract_forehead_skin(frame, face_box)

        if mean_bgr is not None:
            engine.rgb_buffer.append([mean_bgr[2], mean_bgr[1], mean_bgr[0]])

            if len(engine.rgb_buffer) > 300:
                engine.rgb_buffer.pop(0)

            bpm = engine.calculate_bpm()

            if bpm is not None:
                cv2.putText(frame, f"Heart Rate: {bpm:.1f} BPM", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Status: STABLE", (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                csv_writer.writerow([f"{current_time:.2f}", f"{bpm:.1f}"])
            else:
                cv2.putText(frame, "Calibrating (Keep Still)...", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("BioSense Tracker Pro", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
