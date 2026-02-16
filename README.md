# Real-Time Road Anomaly Detection on Arm Edge Hardware
### Bharat AI-SoC Student Challenge | Problem Statement 3

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-red?logo=raspberrypi)](https://www.raspberrypi.com/)
[![Architecture](https://img.shields.io/badge/Arch-Arm%20Cortex--A72-blue?logo=arm)](https://www.arm.com/)
[![Optimization](https://img.shields.io/badge/Optimization-INT8%20Quantization-green)](#optimization-strategy)

---

## Overview

This project implements a high-performance, edge-optimized AI system for detecting road anomalies such as potholes and surface cracks in real time. The system is designed specifically for Raspberry Pi 4 (Arm Cortex-A72) and processes live dashcam footage to generate actionable insights for municipal road maintenance and driver safety.

The primary engineering objective is achieving low inference latency and high throughput on resource-constrained Arm hardware without using external accelerators.

---

## Key Features

- Real-time inference at 11–12 FPS on Raspberry Pi 4 CPU
- Arm NEON SIMD acceleration via XNNPACK delegate
- INT8 post-training quantization
- Multithreaded pipeline architecture
- CSV-based anomaly logging
- Sustained performance under active cooling

---

## Hardware and Software Stack

### Hardware

- **Compute:** Raspberry Pi 4 Model B (4GB RAM)
- **Processor:** Arm Cortex-A72 (64-bit quad-core)
- **Camera:** 1080p USB Dashcam / Webcam
- **Cooling:** Active dual-fan heatsink

### Software

- **OS:** Raspberry Pi OS 64-bit (Debian Bookworm)
- **Runtime:** ai-edge-litert (Google LiteRT optimized for Arm CPUs)
- **Vision Library:** OpenCV 4.10.x
- **Model:** YOLOv5 Nano (custom-trained and quantized)
- **Delegate:** XNNPACK (Arm-optimized CPU execution)

---

## Optimization Strategy

### 1. Lightweight Architecture
YOLOv5 Nano was selected due to its low FLOP count and suitability for embedded deployment.

### 2. Resolution Scaling
Input resolution optimized to 320x320 to reduce computational cost by approximately 4x compared to 640px models while preserving detection accuracy.

### 3. INT8 Post-Training Static Quantization
- Converted model weights from FP32 to INT8
- Reduced model size by approximately 75%
- Enabled Arm-optimized XNNPACK delegate
- Improved inference throughput significantly

### 4. Multithreaded Pipeline
Implemented a thread-safe VideoStream class:
- Separates frame capture from inference
- Eliminates I/O blocking
- Ensures inference engine always receives latest frame

### 5. Thermal Optimization
Active cooling prevents CPU frequency throttling and ensures sustained real-time execution.

---

## Performance Benchmarks (Raspberry Pi 4)

| Metric | Result |
|--------|--------|
| Inference Latency | ~82 ms |
| Throughput | 11.6 FPS |
| Memory Usage | < 150 MB |
| CPU Utilization | ~65% |

Benchmarks measured during continuous live video processing at 320x320 resolution.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐    ┌──────────┐    ┌─────────────────────────┐   │
│   │ Camera  │───▶│  Resize  │───▶│  INT8 Quantized YOLOv5n │   │
│   │ Capture │    │ Normalize│    │  (XNNPACK Delegate)     │   │
│   └─────────┘    └──────────┘    └───────────┬─────────────┘   │
│                                              │                  │
│                                              ▼                  │
│   ┌─────────┐    ┌──────────┐    ┌─────────────────────────┐   │
│   │   CSV   │◀───│  Overlay │◀───│    Post-Processing      │   │
│   │  Logger │    │  Render  │    │  (NMS, Box Extraction)  │   │
│   └─────────┘    └──────────┘    └─────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Entire pipeline runs exclusively on Arm Cortex-A72 CPU cores.

---

## Installation and Usage

### 1. Clone Repository

```bash
git clone https://github.com/Aanjaneya24/Bharat-AI-SoC-Road-Anomaly-Detection.git
cd Bharat-AI-SoC-Road-Anomaly-Detection
```

### 2. Install Dependencies

```bash
sudo apt update
sudo apt install python3-opencv -y
pip3 install ai-edge-litert numpy --break-system-packages
```

### 3. Run Inference

```bash
python final_submission.py
```

---

## Output

The system generates:

**road_anomalies_log.csv**

| Timestamp           | Anomaly_Type | Confidence |
| ------------------- | ------------ | ---------- |
| 2026-02-17 14:02:11 | Pothole      | 0.84       |
| 2026-02-17 14:02:12 | Pothole      | 0.79       |

This output can be integrated into GIS platforms or smart city dashboards.

---

## Demo Requirements

For validation purposes, the demo video should show:

1. Physical Raspberry Pi 4 hardware with cooling visible
2. Terminal running inference
3. On-screen FPS and latency overlay
4. Real-time pothole detection

---

## Engineering Highlights

- CPU-only edge inference
- Arm NEON optimized execution
- INT8 quantized deployment
- Production-ready multithreaded design
- Performance exceeding competition requirement by more than 2x

---

## Acknowledgements

- **Arm and Arm Education**
- **IIT Delhi**
- **Ministry of Electronics and IT (MeitY)**

---

**Author:** Aanjaneya Pandey  
**Date:** February 2026  
**Project:** Bharat AI-SoC Student Challenge
