import streamlit as st
import numpy as np
import os
import torch

# --- Stability Configuration (Critical for Server) ---
# MUST be placed immediately after the torch import to prevent the 
# "cannot set number of interop threads" error. The failure occurs 
# because Streamlit's runtime (or other imports) triggers parallel 
# work before these lines can execute.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import cv2
from ultralytics import YOLO
from PIL import Image

cv2.setNumThreads(0) # Placing this here is safer, though it can remain with the torch settings.

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

# We define a function to encapsulate the heavy synchronous process
def process_image(model, uploaded_file):
    """Handles image loading, resizing, inference, and result display."""
    
    # Create the output directory for saving results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Use a unique name for the saved image to prevent browser caching issues
    unique_id = os.urandom(8).hex()
    input_path = os.path.join(OUTPUT_DIR, f"input_{unique_id}.jpg")
    save_path = os.path.join(OUTPUT_DIR, f"output_{unique_id}.jpg")
    
    # Read the file bytes
    image_bytes = uploaded_file.read()

    # Convert bytes to a numpy array for OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Image Resizing (for stability on slow CPUs)
    # Check if the image needs resizing
    h, w = image.shape[:2]
    long_side = max(h, w)
    
    if long_side > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / long_side
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Save the processed image for YOLO input
    cv2.imwrite(input_path, image)
    
    # 2. Run Inference (The CPU intensive part)
    # The image is already saved, so we can pass the path directly to YOLO
    # Using save=True will save the annotated image directly to runs/detect/predictX/
    # We set conf=0.25 for a general balance, but this can be adjusted.
    
    # Run the prediction
    results = model.predict(
        source=input_path, 
        conf=0.25, 
        iou=0.7, 
        save=True,
        project=OUTPUT_DIR,
        name=unique_id,
        exist_ok=True, # Allow the directory to exist
        verbose=False # Keep the console clean
    )
    
    # 3. Process and Return Results
    
    # The results are saved in a sub-folder created by YOLO. We need to find the correct path.
    # YOLO typically saves to runs/detect/unique_id/
    yolo_output_dir = os.path.join(OUTPUT_DIR, unique_id)
    yolo_output_file = os.path.join(yolo_output_dir, os.path.basename(input_path))
    
    # YOLO saves the annotated image to a specific path; we need to rename it for simplicity
    final_annotated_path = os.path.join(OUTPUT_DIR, f"annotated_{unique_id}.jpg")
    
    # The annotated file is usually named the same as the input file by YOLO, but placed 
    # inside its run directory. We move it to the expected save_path.
    try:
        os.rename(yolo_output_file, final_annotated_path)
    except FileNotFoundError:
        # Handle case where YOLO might save under a different name or structure
        # Fallback: check other files in the output directory
        for f in os.listdir(yolo_output_dir):
            if f.endswith('.jpg') or f.endswith('.png'):
                os.rename(os.path.join(yolo_output_dir, f), final_annotated_path)
                break
        
    # Get the number of detections
    detection_count = 0
    if results and len(results) > 0 and results[0].boxes:
        detection_count = len(results[0].boxes)
        
    return final_annotated_path, detection_count

# --- Streamlit UI and Control Flow ---

st.set_page_config(
    page_title="DrainSight AI Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HF STYLE HEADER AND INTRO (Refined) ---
# Centered Header
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>Drainage Detector 🚧</h1>", unsafe_allow_html=True)

# Instruction Block with Border (HF Style)
with st.container(border=True):
    st.header("⚡️ Defect Detection Instructions")
    st.markdown("""
    Upload a high-resolution image of a drainage channel on the left to begin the analysis.
    
    This application uses a custom **YOLOv8** model to automatically detect and localize 
    common defects such as **silt, plastic waste, and structural cracks**.
    """)
    st.caption("🚨 Note: Due to the shared server environment, analysis may take up to 10 seconds.")


# Two columns for input and results
col_input, col_results = st.columns([1, 1])

# --- Input Column (Left) ---

with col_input:
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png", "webp"],
        key="file_uploader"
    )
    
    # Use a placeholder for the results column to manage state updates cleanly
    results_placeholder = st.empty() 
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.subheader("Uploaded Image")
        st.image(uploaded_file, caption="Input Image", use_column_width=True)
        
        # --- Asynchronous Control Flow ---
        
        # Check if the analysis is already running
        if 'running' not in st.session_state:
            st.session_state.running = False

        if not st.session_state.running:
            # Button to trigger the analysis
            if st.button("Run Defect Detection", type="primary", use_container_width=True):
                # Set running state to true and rerun the script to show the spinner
                st.session_state.running = True
                st.session_state.uploaded_file = uploaded_file # Store file object
                st.rerun()
        else:
            # Show a disabled button while running
            st.button("Analysis in Progress...", disabled=True, use_container_width=True)


# --- Results Column (Right) ---

with col_results:
    # This is the main logic handler that runs after the button click
    
    # Check if the app is currently running an analysis
    if st.session_state.get('running', False):
        # The model loading will only happen once thanks to @st.cache_resource
        model = load_yolo_model()
        
        if model is not None:
            # Use st.spinner to show a responsive loading indicator
            with st.spinner("⚙️ Analyzing image for defects... This may take a moment on the free server."):
                try:
                    # Run the synchronous inference process
                    save_path, detection_count = process_image(model, st.session_state.uploaded_file)
                    
                    # Clear the running state upon completion
                    st.session_state.running = False
                    
                    if detection_count > 0:
                        # Success: Defects found
                        st.subheader("⚠️ Detection Results: Defects Found!")
                        st.warning(f"Identified **{detection_count}** potential defects/blockages.")
                        # Load the saved image for display
                        annotated_image = Image.open(save_path)
                        st.image(annotated_image, caption="Annotated Image with Bounding Boxes", use_column_width=True)
                    else:
                        # Success: Channel Clear
                        st.subheader("✅ Inspection Complete: Channel Clear!")
                        st.success("No defects or blockages were identified in this channel segment.")
                        
                except Exception as e:
                    # Catch and display any final, unhandled error
                    st.session_state.running = False # Clear state even on error
                    st.subheader("🔴 FATAL APPLICATION ERROR")
                    st.error(f"The analysis failed due to a critical error: {e}")
                    st.toast("Analysis failed! Check the console/logs for details.", icon='❌')
                    st.rerun() # Rerun to update the button state
    
    elif st.session_state.get('uploaded_file') is None and uploaded_file is None:
        # Initial state: display instructions
        st.info("Upload an image on the left to begin the analysis.")
        
    elif uploaded_file is None and st.session_state.get('uploaded_file') is not None:
         # State when the user clears the file, ensure the UI resets
         st.session_state.pop('uploaded_file')
         st.session_state.running = False
         st.rerun()
