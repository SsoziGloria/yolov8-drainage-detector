# 1. START: Use the latest stable slim image (Bookworm) to avoid repository errors.
FROM python:3.10-slim-bookworm

# 2. Set the working directory
WORKDIR /app

# 3. CRITICAL FIX: Install essential system utilities and GLIB dependency
# libgthread-2.0-0 is sometimes the final missing system link for the runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    ffmpeg \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. ENFORCEMENT STEP: Explicitly uninstall standard opencv, ensuring only headless remains.
# This prevents ultralytics from pulling in the full, graphics-dependent version that causes the libGL.so.1 error.
# The '|| echo' ensures the build doesn't fail if the package wasn't installed.
RUN pip uninstall -y opencv-python || echo "opencv-python not found, continuing..."

# 6. Copy the model and app code
COPY best.pt .
COPY streamlit_app.py .

# 7. Expose the port
EXPOSE 8501 

# 8. The command that starts the Streamlit web server
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
