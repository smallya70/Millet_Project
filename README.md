# Millet Quality Control Vision Agent (FSSAI MVP)

## Overview
This repository contains a real-time, Edge AI Vision Agent designed for industrial agricultural packaging lines. It acts as an automated quality control inspector to ensure FSSAI compliance by instantly detecting physical cross-contamination (stones, soil, extraneous matter) in continuous grain flows.

The current MVP is trained specifically to differentiate between **Little Millet**, **Bajra**, and **Contamination** using a custom-engineered dataset.

## Business Value
* **Real-Time FSSAI Compliance:** Automatically flags physical hazards before packaging.
* **Edge Processing:** Runs completely locally on edge hardware with zero cloud latency.
* **Enterprise Ready:** Architected to bridge the gap between factory floor hardware (PLCs) and enterprise resource planning (ERP/WMS).

## Technical Architecture
* **Computer Vision:** `OpenCV` for real-time video frame extraction and live UI overlays.
* **Machine Learning:** `TensorFlow` / `Keras` (Legacy Keras Bridge) utilizing a lightweight MobileNet-based architecture trained via Google Teachable Machine.
* **Data Pipeline:** Custom Python extraction scripts transforming raw 30fps MP4 CCTV footage into balanced, classified training frames.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/smallya70/Millet_Project.git
   cd Millet_Project