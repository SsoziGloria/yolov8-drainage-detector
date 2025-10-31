import streamlit as st
import numpy as np
import os
import torch
import cv2
from ultralytics import YOLO
from PIL import Image

# --- Stability Configuration (Critical for Server) ---
# Aggressively limit multi-threading to prevent memory deadlocks and silent crashes
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cv2.setNumThreads(0)

# --- Configuration ---
MODEL_PATH = "best.pt"  # Reverting to your trained model
OUTPUT_DIR = "temp_results"
MAX_IMAGE_SIZE = 1200 # Resize larger images to prevent memory errors

# --- Model Loading (Caching for stability) ---

@st.cache_resource(show_spinner=False)
def load_yolo_model():
    """Loads the YOLO model only once."""
    try:
        # Check if the model file exists, if not, deployment will fail later
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}. Check Git LFS status.")
            return None
        
        # Load the model using the standard YOLO constructor
        model = YOLO(MODEL_PATH)
        return model

    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {e}")
        st.warning("Please verify the integrity of the 'best.pt' file.")
        return None

# --- Main App Logic ---

def process_image(model, uploaded_file):
    """Handles image loading, resizing, inference, and result display."""
    
    # Create the output directory for saving results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load and prepare image
    input_image_pil = Image.open(uploaded_file)
    # Crucial step: Convert to 3-channel RGB to prevent shape mismatch errors
    input_image_pil = input_image_pil.convert("RGB")
    
    # Resize image if it's too large (prevents memory errors)
    width, height = input_image_pil.size
    if max(width, height) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(width, height)
        input_image_pil = input_image_pil.resize((int(width * ratio), int(height * ratio)))
        st.toast(f"Image resized to {input_image_pil.size[0]}x{input_image_pil.size[1]} for stability.", icon='📏')

    # Convert to NumPy array for robust YOLO processing
    input_image_np = np.array(input_image_pil)
    
    # 2. Run Inference
    # The predict method is safer than the call method in deployed environments
    results = model.predict(source=input_image_np, save=False, conf=0.25, iou=0.7)
    
    # 3. Check for Detections
    # results[0].boxes.data.shape[0] gives the count of detected bounding boxes
    detections_found = results[0].boxes.data.shape[0] > 0
    
    # 4. Save results to disk (guaranteed plotting stability)
    save_path = os.path.join(OUTPUT_DIR, f"result_{os.path.basename(uploaded_file.name)}")
    results[0].save(filename=save_path) # Saves the annotated image to disk
    
    return detections_found, save_path, results[0].boxes.data.shape[0]

# --- Streamlit UI Layout ---

st.set_page_config(layout="wide", page_title="YOLOv8 Drainage Detector 🚧")

st.title("YOLOv8 Drainage Detector 🌊")

# --- About Section ---
with st.expander("ℹ️ About the Detector", expanded=False):
    st.markdown("""
    This application uses a custom-trained **YOLOv8** model to analyze drainage channel images for defects, 
    blockages, and integrity issues. This AI system provides crucial, measurable data for commercial and municipal inspection teams.
    """)

# --- Main Content Columns ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Image 🖼️")
    uploaded_file = st.file_uploader(
        "Choose an image of a drainage channel or pipe:", 
        type=['jpg', 'jpeg', 'png']
    )

with col2:
    st.subheader("2. Run Detection")
    # Placeholders for dynamic content
    results_placeholder = st.empty()
    run_button_placeholder = st.empty()

# --- Execution Flow ---
model = load_yolo_model()

if uploaded_file is not None:
    # Display the original image
    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, caption="Image ready for analysis.", use_column_width=True)

    # Check if model is loaded and display the button
    if model is not None:
        if run_button_placeholder.button("🔍 Run Defect Detection"):
            # Execute the processing function inside a spinner
            with st.spinner("Analyzing image for defects..."):
                try:
                    detections_found, save_path, detection_count = process_image(model, uploaded_file)
                    
                    if detections_found:
                        # Success: Defects found
                        results_placeholder.subheader("⚠️ Detection Results: Defects Found!")
                        results_placeholder.warning(f"Identified **{detection_count}** potential defects/blockages.")
                        # Load the saved image for display
                        annotated_image = Image.open(save_path)
                        results_placeholder.image(annotated_image, caption="Annotated Image with Bounding Boxes", use_column_width=True)
                    else:
                        # Success: Channel Clear
                        results_placeholder.subheader("✅ Inspection Complete: Channel Clear!")
                        results_placeholder.success("No defects or blockages were identified in this channel segment.")
                        
                except Exception as e:
                    # Catch and display any final, unhandled error
                    results_placeholder.subheader("🔴 FATAL APPLICATION ERROR")
                    results_placeholder.error(f"The analysis failed due to a critical error: {e}")
                    st.toast("Analysis failed! Check the console/logs for details.", icon='❌')
    else:
        # Model load failed, display error in the results placeholder
        results_placeholder.error("Model failed to load. Check the initial error message above.")
elif uploaded_file is None:
    # Clear the second column if no file is uploaded
    results_placeholder.empty()