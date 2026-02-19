# Project Report: Real-Time Road Anomaly Detection on Arm Edge Hardware
**Bharat AI-SoC Student Challenge | Problem Statement 3**

**Author:** Aanjaneya Pandey  
**Submission Date:** February 20, 2026  
**Target Hardware:** Raspberry Pi 4 Model B (Arm Cortex-A72)  

---

## 1. Introduction
This project aims to solve the problem of road safety and maintenance by detecting road anomalies (potholes and cracks) in real-time. The solution is specifically engineered for resource-constrained edge devices, specifically the Raspberry Pi 4, utilizing the power of the Arm Cortex-A72 processor without the need for additional AI accelerators.

## 2. Methodology & Architecture
The system is built upon a high-efficiency computer vision pipeline designed for low-latency inference.

### 2.1 Model Architecture
- **Base Model:** YOLOv5 Nano (YOLOv5n).
- **Optimization:** INT8 Post-Training Static Quantization.
- **Input Size:** 320x320 pixels.
- **Framework:** LiteRT (formerly TFLite) with XNNPACK acceleration.

### 2.2 Inference Pipeline
The application utilizes a **Multithreaded Pipeline** to maximize CPU utilization:
1.  **Capture Thread:** A dedicated thread continuously grabs frames from the USB camera to prevent I/O blocking.
2.  **Inference Thread:** 
    -   **Preprocessing:** Resizes and converts frames to RGB.
    -   **Model Execution:** Uses the AI-Edge-LiteRT interpreter with XNNPACK delegates for Arm-optimized execution.
    -   **Post-processing:** Filters detections based on confidence thresholds and logs results.

## 3. Hardware Utilization & Optimization
Optimizing for the Arm Cortex-A72 required a multi-layered approach:

### 3.1 Arm NEON & XNNPACK Acceleration
By using the **XNNPACK delegate** through LiteRT, the system leverages **Arm NEON SIMD** instructions. This allows the CPU to perform multiple mathematical operations in parallel, which is critical for the convolutional layers of the YOLO model.

### 3.2 INT8 Quantization
The model was quantized from FP32 to **INT8 weights and activations**. This provides several benefits:
-   **Size Reduction:** The model size is reduced by ~75% (to approx. 2MB).
-   **Inference Speed:** Integer arithmetic is significantly faster than floating-point on the Cortex-A72.
-   **Efficiency:** Lower power consumption and memory bandwidth requirements.

### 3.3 Thermal Management
The system is designed to run with active cooling. This ensures that the CPU maintains its maximum clock frequency (1.5GHz+) during continuous inference, preventing thermal throttling that would otherwise degrade performance.

## 4. Results & Performance
The system achieves sustained real-time performance on a Raspberry Pi 4:

| Metric | Performance |
| :--- | :--- |
| **Average Inference Latency** | ~82 ms |
| **Total Pipeline Throughput** | ~11.6 FPS |
| **Model Size** | ~1.97 MB (INT8) |
| **CPU Utilization** | ~65-70% (distributed across cores) |
| **Memory Footprint** | < 150 MB |

## 5. Deployment & Logging
-   **Real-time Overlay:** The system provides a live video feed with bounding box overlays and performance metrics (FPS/Latency).
-   **CSV Logging:** All detected anomalies are logged to `road_anomalies_log.csv` with precise timestamps and confidence levels, making the data ready for GIS mapping or municipal reporting systems.

## 6. Conclusion
The "Bharat AI-SoC Road Anomaly Detection" system demonstrates that high-performance, real-time AI is achievable on edge Arm hardware through careful model selection and hardware-aware optimization. The project fulfills all requirements of the challenge, providing a cost-effective and scalable solution for infrastructure monitoring.
