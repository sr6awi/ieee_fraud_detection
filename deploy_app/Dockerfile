# ================================================================
# 🐳 IEEE Fraud Detection - Dual Path Model (FastAPI + TensorFlow)
# ================================================================

# 1️⃣ Base image with Python and system deps
FROM python:3.10-slim

# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy requirements first (for better caching)
COPY requirements.txt .

# 4️⃣ Install system packages + Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5️⃣ Copy the app files
COPY . .

# 6️⃣ Expose FastAPI default port
EXPOSE 8000

# 7️⃣ Start the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
