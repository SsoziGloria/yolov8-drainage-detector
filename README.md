# 🌊 YOLOv8 Drainage Defect Detector

This is the repository for the AI-powered drainage inspection system, utilizing YOLOv8 for real-time detection of defects and blockages in channel infrastructure. Developed for the Department of Computer Science, Makerere University, this project aims to provide a reliable, automated tool for infrastructure maintenance and commercial inspection services.

## 💡 Project Overview
The core problem addressed is the inefficient, costly, and often dangerous manual inspection of drainage systems. Our solution leverages computer vision to automatically identify and classify defects (e.g., cracks, blockages, erosion) from captured images.

## 🛠️ Deployment Instructions

This application is configured for robust deployment using Docker on platforms like Streamlit Community Cloud or AWS/Azure.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/SsoziGloria/yolov8-drainage-detector.git](https://github.com/SsoziGloria/yolov8-drainage-detector.git)
    cd yolov8-drainage-detector
    ```

2.  **Initialize Git LFS:** The `best.pt` model file is stored via Git LFS. You must initialize LFS and pull the model:
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Local Run (for testing):**
    ```bash
    pip install -r requirements.txt
    streamlit run streamlit_app.py
    ```

4.  **Cloud Deployment (Recommended):**
    The project includes a `Dockerfile` and `requirements.txt` for one-click deployment via **Streamlit Community Cloud**. Select "Use existing Dockerfile" during setup.

## 📦 File Structure

| File | Description |
| :--- | :--- |
| `streamlit_app.py` | The main Streamlit application script. |
| `Dockerfile` | Custom image build instructions (Python 3.10-slim with CV dependencies). |
| `requirements.txt` | List of all Python packages (including `torch` and `ultralytics`). |
| `.gitattributes` | Git LFS configuration file. |
| `best.pt` | The trained YOLOv8 model weights (tracked via LFS). |