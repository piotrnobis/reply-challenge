# reply-challenge
This is LLMBrew's take on the reply challenge at 2025 Makethon at TUM.

# Setup
Install requirements using the requirements file `pip install -r requirements.txt`

## Overview

This project implements a modular approach to target identification using **object detection** and **object segmentation** techniques. The goal is to detect and identify pallets and barcodes within images, then return their positions relative to the initial camera frame.

## Approach
### 1) Target Identification
The system follows a two-step process:
- **Object Detection (Pallets)**:
   - Apply object detection to obtain bounding boxes that contain **pallets**.
   - A **YOLOv12** model is fine-tuned using both dataset and AI-generated images.
- **Object Segmentation (Barcodes)**:
   - Once the pallets are detected, object segmentation is applied to the cropped image segments to detect **barcodes**.
   - A **YOLOv11** model is fine-tuned using both manually and automatically cropped images, focusing on the barcode (main object) and its foreground.
- **Return Object Positions**:
   - The positions of the detected objects (pallets and barcodes) are returned on a **2D plane** relative to the initial camera frame.

## 2) 3D Reconstruction
- Apply **Structure from Motion (SfM)** to build a 3D scene using features obtained via **SIFT** (Scale-Invariant Feature Transform).
- Match the obtained scene features with those obtained during **target identification** (pallet and barcode positions).
- Perform **3D visualization** using **matplotlib** to view the reconstructed scene.

## 3) Route Planning
- Given the locations of interest with global coordinates, apply a **path planning optimization** algorithm to determine the optimal path and assign waypoints.
- Transform the planned path into the **.kmz** format for use in tools like Google Earth.
