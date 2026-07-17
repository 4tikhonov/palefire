FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (e.g., for OpenCV, though headless avoids most)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    tesseract-ocr \
    poppler-utils \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Default command
CMD ["bash"]
