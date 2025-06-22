FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.11 and pip
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt

# Install llama-cpp-python with OpenBLAS support
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
RUN pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    --no-cache-dir

# Copy the rest of the app
COPY . .

# Default command
CMD ["bash"]
