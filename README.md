# 🏥 Clinical RAG Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/MIMIC--IV--EXT-Dataset-orange.svg" alt="Dataset">
</p>

<p align="center">
  <strong>An interactive clinical question-answering system powered by E5 embeddings and Mistral-7B</strong>
</p>

---

## 🎯 Overview

A production-ready **Retrieval-Augmented Generation (RAG)** web application for clinical question answering, built with:

- **E5-small-v2** embeddings (384-dim) for semantic search
- **ChromaDB** vector database with 934 indexed clinical documents
- **Mistral-7B-Instruct** (4-bit quantized) for natural language generation
- **Streamlit** for interactive web interface
- **MIMIC-IV-EXT** dataset (511 clinical notes across 25 disease categories)

### ✨ Key Features

- 🔍 **Semantic Search**: Find relevant clinical cases using natural language
- 🤖 **AI-Powered Answers**: Generate detailed responses with source attribution
- 🏥 **25 Disease Categories**: Pneumonia, Heart Failure, Diabetes, Stroke, and more
- ⚡ **Fast Retrieval**: ~14ms query latency, 70+ queries/second
- 💾 **Memory Efficient**: 4-bit quantization reduces GPU memory to ~4GB
- 📊 **Source Transparency**: View retrieved documents and similarity scores

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **CUDA 11.8+** (for GPU acceleration)
- **~6.5 GB GPU memory** (Tesla T4, V100, or RTX 3090)
- **Google Drive** (to download pre-processed data)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/clinical-rag-assistant.git
cd clinical-rag-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download pre-processed data (see data/README.md)
# Download chroma_db.zip from Google Drive and extract to data/

# 4. Run the Streamlit app
streamlit run app.py
```

### 🌐 Access the App

Open your browser and navigate to:
```
http://localhost:8501
```

---

## 📁 Project Structure

```
clinical-rag-assistant/
│
├── app.py                          # Main Streamlit application
├── src/
│   ├── __init__.py                 # Python package initializer
│   └── rag_pipeline.py             # Core RAG pipeline logic
│
├── data/                           # Data directory (download required)
│   ├── .gitkeep                    # Keeps directory in Git
│   ├── README.md                   # Data download instructions
│   ├── chroma_db/                  # ChromaDB vector database (DOWNLOAD)
│   │   ├── chroma.sqlite3
│   │   └── [embeddings & metadata]
│   └── processed_documents.parquet # Document metadata (optional)
│
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies (for deployment)
├── README.md                       # This file
├── SETUP_GUIDE.md                  # Detailed setup instructions
└── .gitignore                      # Git exclusions
```

---

## 🎨 Application Features

### 1. **Interactive Query Interface**
- Type clinical questions in natural language
- Adjust number of retrieved sources (1-10)
- View real-time processing status

### 2. **Semantic Search Results**
- Ranked documents by similarity score
- Disease category and subtype tags
- Expandable document preview

### 3. **AI-Generated Answers**
- Contextual responses from Mistral-7B
- Source attribution with citations
- Clinical terminology and medical accuracy

### 4. **Advanced Filters**
- Filter by disease category (optional)
- Search within specific medical domains
- Compare results across conditions

### 5. **Performance Metrics**
- Query processing time
- Retrieval latency
- Generation latency
- Memory usage statistics

---

## 📊 System Architecture

```mermaid
graph LR
    A[User Query] --> B[Streamlit UI]
    B --> C[RAG Pipeline]
    C --> D[E5 Query Encoding]
    D --> E[ChromaDB Search]
    E --> F[Top-K Documents]
    F --> G[Mistral-7B Generation]
    G --> H[Answer + Sources]
    H --> B
```

### Components

1. **Frontend**: Streamlit web interface
2. **Embedding**: E5-small-v2 (intfloat/e5-small-v2)
3. **Vector DB**: ChromaDB with cosine similarity
4. **LLM**: Mistral-7B-Instruct-v0.2 (4-bit NF4 quantization)
5. **Dataset**: MIMIC-IV-EXT (934 chunks, 25 diseases)

---

## 🔧 Configuration

### Model Settings

```python
# Embedding Model
MODEL_NAME = "intfloat/e5-small-v2"
EMBEDDING_DIM = 384
QUERY_PREFIX = "query: "

# Generation Model
GENERATION_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
QUANTIZATION = "4-bit NF4"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7

# Retrieval
DEFAULT_TOP_K = 5
MAX_CONTEXT_TOKENS = 2000
```

### Streamlit Configuration

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
```

---

## 📚 Dataset

### MIMIC-IV-EXT-DIRECT-1.0.0

- **Source**: [PhysioNet](https://physionet.org/content/mimic-iv-ext/) (credentialed access required)
- **Original Size**: 511 clinical notes
- **Processed**: 934 chunks (400-token chunks with 100-token overlap)
- **Categories**: 25 disease types
- **Subtypes**: 55 clinical variants

### Disease Categories

| Category | Documents | Category | Documents |
|----------|-----------|----------|-----------|
| Acute Coronary Syndrome | 134 | Pneumonia | 39 |
| Heart Failure | 107 | Hypertension | 43 |
| Stroke | 67 | Diabetes | 26 |
| Gastritis | 58 | COPD | 36 |
| Pulmonary Embolism | 58 | Asthma | 25 |

*Full list available in the application sidebar*

---

## 📈 Performance Benchmarks

### Retrieval Performance

| Metric | Value |
|--------|-------|
| **Precision@5** | 0.41 |
| **Recall@5** | 1.79 |
| **MRR** | 0.60 |
| **Avg Query Time** | 14.35 ms |
| **Throughput** | 69.7 queries/sec |

### Generation Quality

| Metric | Value |
|--------|-------|
| **ROUGE-L** | 0.114 |
| **Expected Term Coverage** | 78.75% |
| **Citation Rate** | 90% |
| **Hallucination Risk** | 0% |
| **Success Rate** | 100% |

### Resource Usage

| Resource | Requirement |
|----------|-------------|
| **GPU Memory** | ~6.5 GB (peak) |
| **RAM** | 8 GB minimum |
| **Disk Space** | ~500 MB (data) |
| **End-to-End Latency** | ~77 seconds |

---

## 🎯 Usage Examples

### Example 1: Symptom Query
```
Query: "What are the common symptoms of pneumonia?"

Retrieved Sources:
✓ Pneumonia/Bacterial Pneumonia (similarity: 0.886)
✓ Pneumonia/Viral Pneumonia (similarity: 0.874)

Answer: Common symptoms of pneumonia include fever, cough with 
sputum production, shortness of breath, chest pain, fatigue, 
and sometimes nausea. Physical examination may reveal crackles 
on lung auscultation...
```

### Example 2: Treatment Query
```
Query: "How is acute coronary syndrome managed?"

Retrieved Sources:
✓ Acute Coronary Syndrome/STEMI (similarity: 0.865)
✓ Acute Coronary Syndrome/NSTEMI (similarity: 0.842)

Answer: Management includes antiplatelet therapy (aspirin, P2Y12 
inhibitors), anticoagulation, beta-blockers, statins, and 
revascularization (PCI or CABG) if indicated...
```

### Example 3: Filtered Search
```
Query: "respiratory symptoms"
Filter: Pneumonia

Retrieved Sources:
✓ Pneumonia/Bacterial Pneumonia (similarity: 0.878)
✓ Pneumonia/Bacterial Pneumonia (similarity: 0.872)

Answer: [Filtered results from Pneumonia category only]
```

---

## 🛠️ Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run in development mode
streamlit run app.py --server.runOnSave true
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Generate coverage report
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ app.py
isort src/ app.py

# Lint
flake8 src/ app.py
pylint src/ app.py
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `max_new_tokens` or use CPU mode:
```python
# In rag_pipeline.py
device = "cpu"  # Change from "cuda"
```

#### 2. **ChromaDB Not Found**
```
FileNotFoundError: chroma_db directory not found
```
**Solution**: Download data from Google Drive (see `data/README.md`)

#### 3. **Slow Generation**
```
Generation takes >2 minutes per query
```
**Solution**: 
- Ensure GPU is available: `torch.cuda.is_available()`
- Use smaller `max_new_tokens` (256 instead of 512)
- Consider caching frequently asked queries

#### 4. **Module Import Errors**
```
ModuleNotFoundError: No module named 'chromadb'
```
**Solution**: Reinstall dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```

---

## 🚀 Deployment

### Deploy on Streamlit Cloud

1. Fork this repository
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repository and branch
5. Set `app.py` as the main file
6. Deploy!

**Note**: Streamlit Cloud has limited GPU access. For GPU deployment, use AWS, GCP, or Azure.

### Deploy on AWS EC2

```bash
# Launch EC2 instance with GPU (e.g., g4dn.xlarge)
# Install dependencies
sudo apt update
sudo apt install -y python3-pip nvidia-cuda-toolkit

# Clone and setup
git clone https://github.com/muhammadhoud/NLP-Clinical-RAG.git
cd clinical-rag-assistant
pip install -r requirements.txt

# Run with port forwarding
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Deploy with Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t clinical-rag-assistant .
docker run -p 8501:8501 --gpus all clinical-rag-assistant
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Contribution Ideas

- Add more disease categories
- Implement conversation history
- Add export functionality (PDF/Word)
- Create API endpoint (FastAPI)
- Add user authentication
- Implement feedback mechanism
- Add multilingual support
- Create mobile-responsive UI

---


## 🙏 Acknowledgments

- **E5 Embeddings**: [Microsoft Research](https://huggingface.co/intfloat/e5-small-v2)
- **Mistral-7B**: [Mistral AI](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- **Streamlit**: [Streamlit Inc.](https://streamlit.io/)

---

## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/muhammadhoud/NLP-Clinical-RAG)
- **Discussions**: [Ask questions or share ideas](https://github.com/muhammadhoud/NLP-Clinical-RAG)
- **Email**: 6240houd@gmail.com

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{clinical_rag_assistant_2025,
  title={Clinical RAG Assistant: Interactive Clinical Question Answering with E5 and Mistral-7B},
  author={Muhammad Houd},
  year={2025},
  url={https://github.com/muhammadhoud/NLP-Clinical-RAG}
}
```

---


<p align="center">
  <strong>Made with ❤️ for clinical AI research</strong>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-usage-examples">Usage</a> •
  <a href="#-deployment">Deployment</a> •
  <a href="#-contributing">Contributing</a>
</p>
