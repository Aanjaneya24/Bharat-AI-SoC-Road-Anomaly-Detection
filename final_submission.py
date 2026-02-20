import cv2
import numpy as np
import time
import threading
import os
import ai_edge_litert.interpreter as tflite

# --- CONFIGURATION ---
MODEL_PATH = "best-int8.tflite"
INPUT_SIZE = 320
CONF_THRESHOLD = 0.15  # Ultra-sensitive for laptop screen testing
LOG_FILE = "anomaly_logs.csv"
OUTPUT_VIDEO = "final_pothole_demo.mp4"

# --- MULTITHREADED CAMERA PIPELINE ---
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
print("🚀 Initializing Bharat AI Optimized Engine...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- START CAMERA & RECORDER ---
vs = VideoStream(src=0).start()
time.sleep(1.0) # Wait for camera to adjust
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 10.0, (640, 480))

# Create log file with header
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp,Status,Max_Confidence\n")

print(f"✅ Recording started. Saving to {OUTPUT_VIDEO}")
start_time = time.time()
prev_time = time.time()

try:
    while (time.time() - start_time) < 45: # Record 45s for a solid demo
        frame = vs.read()
        if frame is None: break
        
        display_frame = frame.copy()
        h, w, _ = display_frame.shape

        # 1. Preprocessing (UINT8 for INT8 Model)
        input_img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_img, axis=0).astype(np.uint8)

        # 2. Inference
        inf_start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Dequantize output to float for bounding box math
        output_data = interpreter.get_tensor(output_details[0]['index'])[0].astype(np.float32)
        if np.max(output_data) > 1.0:
            output_data /= 255.0
        
        latency = (time.time() - inf_start) * 1000

        # 3. Detection Logic
        anomaly_found = False
        max_conf = 0.0
        
        for det in output_data:
            confidence = det[4]
            if confidence > max_conf: max_conf = confidence
            
            if confidence > CONF_THRESHOLD:
                anomaly_found = True
                
                # Rescale coordinates to 640x480
                box_w, box_h = det[2] * w, det[3] * h
                x_c, y_c = det[0] * w, det[1] * h
                xmin, ymin = int(x_c - box_w/2), int(y_c - box_h/2)
                xmax, ymax = int(x_c + box_w/2), int(y_c + box_h/2)

                # Draw Visuals (Red Bounding Box)
                cv2.rectangle(display_frame, (max(1,xmin), max(1,ymin)), (min(w-1,xmax), min(h-1,ymax)), (0, 0, 255), 2)
                
                # Label Background
                label = f"POTHOLE: {int(confidence*100)}%"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display_frame, (xmin, ymin - lh - 10), (xmin + lw, ymin), (0, 0, 255), -1)
                cv2.putText(display_frame, label, (xmin, ymin - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 4. HUD DASHBOARD (Top-Left Overlay)
        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        
        # Create a semi-transparent black background
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)

        status_text = "ANOMALY DETECTED" if anomaly_found else "ROAD CLEAR"
        status_color = (0, 0, 255) if anomaly_found else (0, 255, 0)

        cv2.putText(display_frame, f"FPS: {fps:.2f}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"LATENCY: {latency:.1f}ms", (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display_frame, status_text, (15, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # 5. LOGGING & SAVING
        if anomaly_found:
            with open(LOG_FILE, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')},DETECTION,{max_conf:.2f}\n")

        out.write(display_frame)
        # Live Terminal Stats
        print(f"[{status_text}] Max Conf: {max_conf:.2f} | FPS: {fps:.1f} | Time: {int(45-(time.time()-start_time))}s", end='\r')

except KeyboardInterrupt:
    pass

vs.stop()
out.release()
print(f"\n✅ SUCCESS! Demo saved as {OUTPUT_VIDEO}")
