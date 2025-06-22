# 📰 Financial News Summarizer

A simple application for summarizing financial news using Streamlit and machine learning models.

## 🚀 Quick Start with Docker

### Prerequisites
- Docker installed on your system
- GGUF models (optional) - place them in the `data/models/` directory

### Option 1: Docker Compose (Recommended)

1. **Create the models directory:**
```bash
mkdir -p data/models
# Place your GGUF models in the data/models/ directory
```

2. **Build and run with docker-compose:**
```bash
docker-compose up -d
```

3. **Access the application:**
Open your browser and go to: http://localhost:8501

### Option 2: Manual Docker Commands

1. **Build the Docker image:**
```bash
docker build -t news-summarizer .
```

2. **Run the container:**
```bash
docker run -d -p 8501:8501 -v "/path/to/your/gguf/models:/app/data/models" news-summarizer
```

**Example with local models:**
```bash
docker run -d -p 8501:8501 -v "/Users/username/models:/app/data/models" news-summarizer
```

3. **Access the application:**
Open your browser and go to: http://localhost:8501

## 🛠️ Local Development

### Prerequisites
- Python 3.11+
- pip

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the application:**
```bash
streamlit run app.py
```

3. **Access the application:**
Open your browser and go to: http://localhost:8501

## 📋 Features

- **Text Input Field** - Paste your article text for summarization
- **Model Selection** - Choose between GGUF models or simple fallback
- **Summarize Button** - Process the text and generate summary
- **Model Results Block** - Display the summarized content
- **Additional Statistics** - Text length, word count, processing time
- **GGUF Model Support** - Mount local GGUF models for inference

## 🏗️ Architecture

- **Frontend**: Streamlit web interface
- **Backend**: Python with transformers and torch
- **Models**: GGUF format models (mountable via Docker volumes)
- **Container**: Docker with Ubuntu 22.04 base

## 📁 Project Structure

```
financial-news-summarizer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── data/
│   └── models/           # Local GGUF models directory (mounted to container)
├── src/                  # Source code
├── scripts/              # Utility scripts
└── notebooks/            # Jupyter notebooks
```

## 🔧 Configuration

### Docker Volume Mounting
- Mount your local GGUF models to `/app/data/models` in the container
- Models will be automatically detected and available for inference
- Supported format: `.gguf` files

### Environment Variables
- `CMAKE_ARGS`: Configure llama-cpp-python compilation
- `DEBIAN_FRONTEND`: Set to noninteractive for Docker builds
- `STREAMLIT_SERVER_PORT`: Streamlit server port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Streamlit server address (default: 0.0.0.0)

### Model Integration
The application currently includes:
- **Simple Fallback**: Basic text processing without ML models
- **GGUF Model Support**: Placeholder for llama-cpp-python integration
- **Model Detection**: Automatic discovery of mounted GGUF files

## 🚀 Deployment

The application is containerized and ready for deployment on:
- Local Docker environment
- Cloud platforms (AWS, GCP, Azure)
- Kubernetes clusters
- Docker Swarm

## 📝 License

This project is open source and available under the MIT License.
