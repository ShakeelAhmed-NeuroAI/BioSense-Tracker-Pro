import cv2
import numpy as np
import csv
import time
import urllib.request
import os
from scipy.signal import butter, filtfilt

# Ensure Haar Cascade XML exists locally
xml_filename = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_filename):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, xml_filename)

face_cascade = cv2.CascadeClassifier(xml_filename)

class StableBioSenseEngine:
    def __init__(self, window_seconds=5.0):
        self.window_seconds = window_seconds
        self.rgb_buffer = []
        self.time_buffer = []
        self.smoothed_bpm = None
        self.prev_face_box = None

    def smooth_face_box(self, new_box, alpha=0.85):
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
        x, y, w, h = self.smooth_face_box(face_box)
        
        # Lock strictly to the upper forehead center
        fh_y1 = int(y + h * 0.12)
        fh_y2 = int(y + h * 0.28)
        fh_x1 = int(x + w * 0.38)
        fh_x2 = int(x + w * 0.62)

        # Boundary checks
        fh_y1, fh_y2 = max(0, fh_y1), min(frame.shape[0], fh_y2)
        fh_x1, fh_x2 = max(0, fh_x1), min(frame.shape[1], fh_x2)

        forehead_roi = frame[fh_y1:fh_y2, fh_x1:fh_x2]
        if forehead_roi.size == 0:
            return None

        cv2.rectangle(frame, (fh_x1, fh_y1), (fh_x2, fh_y2), (0, 255, 0), 2)
        mean_val = cv2.mean(forehead_roi)[:3]
        return mean_val  # (B, G, R)

    def pos_algorithm(self, rgb_signals):
        RGB = np.array(rgb_signals)
        mean_rgb = np.mean(RGB, axis=0) + 1e-6
        norm_rgb = RGB / mean_rgb

        S1 = norm_rgb[:, 1] - norm_rgb[:, 2]
        S2 = norm_rgb[:, 1] + norm_rgb[:, 2] - 2 * norm_rgb[:, 0]
        
        return S1 + (np.std(S1) / (np.std(S2) + 1e-6)) * S2

    def bandpass_filter(self, signal, fps):
        nyq = 0.5 * fps
        low = np.clip(0.83 / nyq, 0.01, 0.98) # ~50 BPM
        high = np.clip(2.5 / nyq, low + 0.01, 0.99) # ~150 BPM
        b, a = butter(2, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def calculate_bpm(self):
        duration = self.time_buffer[-1] - self.time_buffer[0]
        if duration < 3.5:
            return self.smoothed_bpm

        effective_fps = len(self.time_buffer) / duration
        if effective_fps < 10:
            return self.smoothed_bpm

        bvp = self.pos_algorithm(self.rgb_buffer)
        filtered_bvp = self.bandpass_filter(bvp, effective_fps)

        fft_data = np.abs(np.fft.rfft(filtered_bvp))
        freqs = np.fft.rfftfreq(len(filtered_bvp), 1.0 / effective_fps)
        
        valid_idx = np.where((freqs >= 0.83) & (freqs <= 2.5))[0]
        if len(valid_idx) == 0:
            return self.smoothed_bpm

        raw_bpm = freqs[valid_idx[np.argmax(fft_data[valid_idx])]] * 60.0

        # Apply Exponential Moving Average (EMA) to prevent sharp numerical jumps
        if self.smoothed_bpm is None:
            self.smoothed_bpm = raw_bpm
        else:
            # 85% previous value weight + 15% new sample weight
            self.smoothed_bpm = 0.85 * self.smoothed_bpm + 0.15 * raw_bpm

        return self.smoothed_bpm


# Main execution
cap = cv2.VideoCapture(0)
engine = StableBioSenseEngine(window_seconds=5.0)

csv_file = open("pulse_telemetry_log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp_s", "Heart_Rate_BPM"])

print("--- Starting BioSense Tracker Pro ---")
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time() - start_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        face_box = faces[0]
        mean_bgr = engine.extract_forehead_skin(frame, face_box)

        if mean_bgr is not None:
            # Convert BGR to RGB
            engine.rgb_buffer.append([mean_bgr[2], mean_bgr[1], mean_bgr[0]])
            engine.time_buffer.append(current_time)

            while engine.time_buffer and (current_time - engine.time_buffer[0]) > 5.0:
                engine.rgb_buffer.pop(0)
                engine.time_buffer.pop(0)

            bpm = engine.calculate_bpm()

            if bpm is not None:
                cv2.putText(frame, f"Heart Rate: {bpm:.1f} BPM", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, "Signal: LOCKED", (30, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                csv_writer.writerow([f"{current_time:.2f}", f"{bpm:.1f}"])
            else:
                cv2.putText(frame, "Calibrating (Keep Still)...", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("BioSense Tracker Pro", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
