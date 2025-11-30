# Scratch Detection on Text Images (Segmentation Approach)
This is a Tool used to detect scratches on the numbers given by Mowito.ai

This repository contains the implementation of a segmentation-based system for detecting scratches on text images.  
The model uses pixel-level binary masks for bad images and predicts scratch regions using a U-Net model with an EfficientNet encoder.  
From the predicted mask, we derive image-level good/bad classification and bounding boxes for scratches.

---

##  Key Features
- **Segmentation model (U-Net + EfficientNet-B0)**
- **Predicts binary scratch mask**
- **Automatically generates bounding boxes**
- **Image-level classification via mask area threshold**
- **Supports custom threshold values**
- **Training, validation, and inference scripts included**
- **Robust augmentation pipeline (Albumentations)**

---
