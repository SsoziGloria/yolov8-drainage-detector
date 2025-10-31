# Use the official Python image as the base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install system dependencies needed for computer vision (like OpenCV dependencies)
# Install git and git-lfs for fetching the model
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files (including streamlit_app.py and best.pt)
COPY . .

# Initialize Git LFS inside the container to ensure the model is downloaded/available
# This is a critical step for deployment
RUN git lfs pull

# Define the command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]