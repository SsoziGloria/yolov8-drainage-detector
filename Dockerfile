# Use a pre-built PyTorch image. This image is based on Ubuntu and has PyTorch and Python installed.
# Using a specific version ensures stability.
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install necessary system dependencies
# 1. git and git-lfs for model fetching
# 2. libgl1, libsm6, libxext6: CRITICAL for resolving "libGL.so.1" errors when using OpenCV in a headless environment.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install only the application packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Initialize Git LFS inside the container to ensure the model is downloaded/available
RUN git lfs pull

# Define the command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]