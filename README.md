# Surface Defect Segmentation & Classification
An end-to-end Computer Vision pipeline designed to detect, localize, and classify surface scratches with pixel-level precision. This system utilizes a U-Net architecture with an EfficientNet-B0 backbone to solve the sparse-feature detection problem in industrial quality control.

---
# System Architecture

The pipeline transforms raw sensor data into actionable defect metadata through three primary stages:

Stochastic Augmentation: Handling environmental noise via Albumentations.

Neural Inference: Pixel-wise probability mapping using Encoder-Decoder logic.

Geometric Post-Processing: Converting masks into vectorized area metrics and Bounding Boxes.

---

# Key Features 
Architectural Hybrid: Combines the localization power of U-Net with the feature-extraction efficiency of EfficientNet-B0.

Vectorized Post-Processing: Automatically generates Bounding Boxes and calculates Defect Area Ratios using optimized OpenCV contours.

Loss Function Engineering: Employs a BCE + Dice Loss hybrid to combat extreme class imbalance (small scratches vs. large backgrounds).

Production-Ready Inference: Modular scripts support custom sensitivity thresholds and real-time visualization.

---

# Installation & Usage
1. Clone & Setup
```
git clone https://github.com/Uchihakamal1816/Scratch_Detector.git
cd Scratch_Detector
pip install -r requirements.txt
```
3. Training
The training script utilizes an AdamW optimizer and a Cosine Annealing scheduler for stable convergence.
```
python train.py --epochs 50 --batch_size 16 --lr 1e-4
```
3. Inference
Run inference on a single image or a directory to generate masks and area-based classification.
```
python infer.py --source ./test_images --threshold 0.5
```
## Training Logs (20 Epochs)
<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/91470b9b-a007-4e69-ae74-7ab28d44c371" />

## Best model: Epoch 18 with Val Dice = 0.1648

## Results

| Main Image | Output Image |
|-----------|--------------|
| <img width="285" height="504" src="https://github.com/user-attachments/assets/be80040d-9fa5-4197-a5a3-eb15ccc197fa" /> | <img width="285" height="504" src="https://github.com/user-attachments/assets/300e5d2d-51af-4d24-ad6d-e4ac5e612114" /> |


