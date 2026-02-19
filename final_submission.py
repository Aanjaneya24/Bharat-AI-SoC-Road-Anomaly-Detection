import cv2
import numpy as np
import time
import os
import threading
import ai_edge_litert.interpreter as tflite

# --- CONFIGURATION ---
MODEL_PATH = "best-int8.tflite"
INPUT_SIZE = 320
CONF_THRESHOLD = 0.45
LOG_FILE = "road_anomalies_log.csv"

# --- MULTITHREADED CAMERA CLASS ---
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- INITIALIZE AI ---
print("🚀 Launching Bharat AI Optimized Inference Engine...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- INITIALIZE CSV LOGGING ---
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp,Anomaly_Type,Confidence\n")

# --- MAIN INFERENCE LOOP ---
vs = VideoStream(src=0).start() # Change 0 to "test_road.mp4" for file testing
time.sleep(2.0) # Warm up camera

fps_count = 0
start_time = time.time()
print("✅ System Operational. Monitoring Road Conditions...")

try:
    while True:
        frame = vs.read()
        if frame is None: break

        # 1. Optimized Preprocessing
        # Resize and convert to UINT8 for INT8 Model
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img, axis=0).astype(np.uint8)

        # 2. Inference (Optimized for Arm NEON)
        inf_start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        latency = (time.time() - inf_start) * 1000

        # 3. Smart Detection & Logging
        detected = False
        for det in output_data:
            if det[4] > CONF_THRESHOLD:
                detected = True
                # Draw Box (Scaled back to frame size)
                h, w, _ = frame.shape
                cv2.putText(frame, "⚠️ ROAD ANOMALY", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                break

        if detected:
            with open(LOG_FILE, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},Pothole,{det[4]:.2f}\n")

        # 4. Performance Overlay
        fps_count += 1
        elapsed = time.time() - start_time
        fps = fps_count / elapsed
        
        cv2.putText(frame, f"Arm optimized | FPS: {fps:.1f} | Latency: {latency:.1f}ms", 
                    (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 5. Show Output (Use HDMI monitor or VNC)
        cv2.imshow("Bharat AI SoC Challenge", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nShutting down...")

vs.stop()
cv2.destroyAllWindows()