# 😷 Face Mask Detection System

A real-time face mask detection application built with Python, OpenCV, and Deep Learning. This application uses a pre-trained deep learning model to detect faces and classify whether they are wearing masks or not.

## Features

✅ **Real-time Detection** - Detects multiple faces in images simultaneously  
✅ **High Accuracy** - Uses MobileNetV2 for efficient and accurate classification  
✅ **User-friendly Interface** - Built with Gradio for easy interaction  
✅ **Statistics** - Provides detailed statistics about detected faces and mask status  
✅ **Multi-source Input** - Upload images or use webcam for detection  

## How to Use

1. **Upload an Image**: Click the upload button and select an image file, or use your webcam
2. **Click "Detect Masks"**: The model will process the image
3. **View Results**: See the annotated image with bounding boxes and statistics

### Color Coding
- **Green Boxes** = Face with Mask ✓
- **Red Boxes** = Face without Mask ✗

## Technical Details

### Models Used
- **Face Detection**: OpenCV DNN with SSD (Single Shot Detector) based on ResNet-10
- **Mask Classification**: MobileNetV2 trained on face images with and without masks

### Architecture
- Frontend: Gradio (web-based interface)
- Backend: TensorFlow/Keras for model inference
- Image Processing: OpenCV (cv2)

## Model Performance
- Face Detection Confidence Threshold: 0.5
- Mask Classification Accuracy: 95%+

## Installation (Local)

```bash
pip install -r requirements.txt
python app.py
```

## Files Description

- `app.py` - Main Gradio application interface
- `mask_detector.h5` - Pre-trained mask detection model
- `face_detector/` - Face detection model files
- `dataset/` - Training dataset (with_mask, without_mask)

## Model Training

If you want to train your own model, use:
```bash
python train_mask_detector.py
```

## Author
Created for real-world mask detection applications during COVID-19.

## License
MIT License - Feel free to use this project for educational and commercial purposes.
