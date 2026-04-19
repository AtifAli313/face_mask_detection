# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import gradio as gr
import numpy as np
import cv2
import os

# Load models once at startup
def load_models():
    """Load face detector and mask detector models"""
    print("[INFO] Loading models...")
    
    # load face detector model
    prototxtPath = r"face_detector\deploy.prototxt"
    weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    # load mask detector model
    try:
        maskNet = tf.keras.models.load_model("mask_detector.h5")
    except:
        print("[INFO] Loading model as SavedModel format...")
        maskNet = tf.saved_model.load("mask_detector.h5")
    
    print("[INFO] Models loaded successfully!")
    return faceNet, maskNet

def detect_and_predict_mask(frame, faceNet, maskNet):
    """Detect faces and predict mask status"""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

def process_image(image, faceNet, maskNet):
    """Process image and return annotated result with statistics"""
    if image is None:
        return None, "Please upload an image"
    
    # Convert PIL image to OpenCV format
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Detect faces and predict masks
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    # Statistics
    with_mask = 0
    without_mask = 0
    
    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll draw the
        # bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        if label == "Mask":
            with_mask += 1
        else:
            without_mask += 1

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Convert back to RGB for Gradio display
    output_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create statistics message
    total_faces = with_mask + without_mask
    if total_faces == 0:
        stats = "No faces detected in the image."
    else:
        stats = f"""
        **Detection Results:**
        - Total Faces: {total_faces}
        - With Mask: {with_mask} ({with_mask/total_faces*100:.1f}%)
        - Without Mask: {without_mask} ({without_mask/total_faces*100:.1f}%)
        """
    
    return output_image, stats

def create_gradio_interface():
    """Create and return the Gradio interface"""
    # Load models
    faceNet, maskNet = load_models()
    
    # Create process function with models bound
    def process(image):
        return process_image(image, faceNet, maskNet)
    
    # Create Gradio interface
    with gr.Blocks(title="Face Mask Detection") as demo:
        gr.Markdown("# 😷 Face Mask Detection System")
        gr.Markdown("Upload an image to detect faces and identify whether they are wearing masks.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    sources=["upload", "webcam"]
                )
                submit_btn = gr.Button("Detect Masks", variant="primary")
            
            with gr.Column():
                image_output = gr.Image(label="Detected Faces")
        
        stats_output = gr.Markdown("Upload an image to get started")
        
        # Connect the button
        submit_btn.click(
            fn=process,
            inputs=image_input,
            outputs=[image_output, stats_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)
