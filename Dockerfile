# 1. START: Use the latest stable slim image (Bookworm) to avoid repository errors.
FROM python:3.10-slim-bookworm

# 2. Set the working directory
WORKDIR /app

# 3. CRITICAL FIX: Install ALL missing system libraries required by OpenCV/ultralytics
# This comprehensive list ensures all run-time dependencies for graphics,
# image, and video handling (often needed indirectly by cv2) are met,
# permanently fixing the "libGL.so.1" error.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libice6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the model and app code
COPY best.pt .
COPY streamlit_app.py .

# 6. Expose the port
EXPOSE 8501 

# 7. The command that starts the Streamlit web server
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
