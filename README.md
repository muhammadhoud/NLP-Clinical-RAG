# Clinical RAG Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/MIMIC--IV--EXT-Dataset-orange.svg" alt="Dataset">
</p>

<p align="center">
  <strong>A production-ready Retrieval-Augmented Generation (RAG) system for clinical question answering</strong>
</p>

> ⚠️ **DISCLAIMER**: This is a research/educational tool demonstrating RAG architecture for clinical text. It is NOT intended for medical diagnosis, treatment decisions, or patient care. Always consult qualified healthcare professionals for medical advice.

---

## 📋 Overview

An end-to-end RAG application showcasing modern techniques for semantic search and context-aware generation in the clinical domain. Built with production-grade components and deployed as an interactive web application.

**Technical Stack:**
- **E5-small-v2** embeddings (384-dim) for semantic search
- **ChromaDB** vector database with 934 indexed clinical documents
- **Mistral-7B-Instruct** (4-bit quantized) for natural language generation
- **Streamlit** for interactive web interface
- **MIMIC-IV-EXT** dataset (511 clinical notes, 25 disease categories)

### ✨ Key Technical Features

- **Semantic Search Engine** - E5 embeddings with cosine similarity retrieval
- **Efficient Vector Storage** - ChromaDB with optimized indexing (14ms query latency)
- **Memory-Optimized LLM** - 4-bit quantization reduces GPU memory to ~4GB
- **Production Web Interface** - Streamlit app with real-time processing
- **Comprehensive Logging** - Performance metrics and source attribution
- **Scalable Architecture** - Modular design supporting expansion

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **CUDA 11.8+** (for GPU acceleration)
- **~6.5 GB GPU memory** (Tesla T4, V100, or RTX 3090)
- **Google Drive access** (to download pre-processed data)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/muhammadhoud/NLP-Clinical-RAG.git
cd clinical-rag-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download pre-processed data
# See data/README.md for Google Drive link
# Extract chroma_db.zip to data/

# 4. Run the application
streamlit run app.py
```

### Access the App
```
http://localhost:8501
```

---

## 🏗️ System Architecture

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

### Core Components

1. **Embedding Model**: E5-small-v2 (intfloat/e5-small-v2)
   - 384-dimensional dense vectors
   - Optimized for semantic similarity
   - Query prefix: "query: "

2. **Vector Database**: ChromaDB
   - 934 pre-indexed document chunks
   - Cosine similarity search
   - Average query time: ~14ms

3. **Generation Model**: Mistral-7B-Instruct-v0.2
   - 4-bit NF4 quantization
   - Context window: 2000 tokens
   - Temperature: 0.7 for balanced outputs

4. **Dataset**: MIMIC-IV-EXT
   - 511 clinical notes
   - 25 disease categories
   - 400-token chunks with 100-token overlap

---

## 📊 Dataset

### MIMIC-IV-EXT-DIRECT-1.0.0

- **Source**: [PhysioNet](https://physionet.org/content/mimic-iv-ext/) (credentialed access required)
- **License**: PhysioNet Credentialed Health Data License
- **Processing**: 511 notes → 934 chunks (400 tokens, 100 overlap)

### Disease Coverage (Top 10)

| Category | Documents | Category | Documents |
|----------|-----------|----------|-----------|
| Acute Coronary Syndrome | 134 | Pneumonia | 39 |
| Heart Failure | 107 | Hypertension | 43 |
| Stroke | 67 | COPD | 36 |
| Gastritis | 58 | Diabetes | 26 |
| Pulmonary Embolism | 58 | Asthma | 25 |

*Full list includes 25 categories and 55 clinical subtypes*

---

## 📈 Performance Benchmarks

### Retrieval Performance
| Metric | Value | Description |
|--------|-------|-------------|
| **Precision@5** | 0.41 | Relevant docs in top 5 |
| **Recall@5** | 1.79 | Coverage of relevant docs |
| **MRR** | 0.60 | Mean reciprocal rank |
| **Query Latency** | 14.35 ms | Average retrieval time |
| **Throughput** | 69.7 queries/sec | System capacity |

### Generation Metrics
| Metric | Value | Notes |
|--------|-------|-------|
| **ROUGE-L** | 0.114 | Lexical overlap with reference |
| **Term Coverage** | 78.75% | Expected medical terms present |
| **Source Attribution** | 90% | Responses citing retrieved docs |

**Note on ROUGE-L**: The 11.4% score reflects that generated answers often provide different but valid explanations compared to reference text. This is common in medical QA where multiple valid formulations exist. Manual evaluation showed clinically relevant responses, though formal medical expert validation is recommended for production use.

### Resource Usage
| Resource | Requirement |
|----------|-------------|
| **GPU Memory** | ~6.5 GB (peak) |
| **RAM** | 8 GB minimum |
| **Storage** | ~500 MB (data + model) |
| **End-to-End Latency** | ~5-10 seconds per query |

---

## 💻 Application Features

### 1. Interactive Query Interface
- Natural language question input
- Adjustable retrieval count (1-10 sources)
- Real-time processing indicators
- Clear error messaging

### 2. Semantic Search Display
- Ranked documents by similarity score
- Disease category and subtype tags
- Expandable document previews
- Similarity score visualization

### 3. Context-Aware Generation
- Answers grounded in retrieved documents
- Source citation with document IDs
- Medical terminology preservation
- Configurable generation parameters

### 4. Category Filtering (Optional)
- Filter by specific disease categories
- Compare across related conditions
- Domain-specific search refinement

### 5. Performance Monitoring
- Query processing time breakdown
- Retrieval vs generation latency
- Memory usage tracking
- System health indicators

---

## 🎯 Usage Examples

### Example 1: Symptom Inquiry
```
Query: "What are the common symptoms of pneumonia?"

Retrieved Sources:
✓ Pneumonia/Bacterial Pneumonia (similarity: 0.886)
✓ Pneumonia/Viral Pneumonia (similarity: 0.874)

Generated Answer:
Common symptoms of pneumonia typically include fever, productive 
cough with sputum, shortness of breath, chest pain (pleuritic), 
and fatigue. Physical examination may reveal crackles on lung 
auscultation and increased respiratory rate...

[Sources: Documents 234, 245]
```

### Example 2: Treatment Protocol
```
Query: "How is acute coronary syndrome managed in emergency settings?"

Retrieved Sources:
✓ Acute Coronary Syndrome/STEMI (similarity: 0.865)
✓ Acute Coronary Syndrome/NSTEMI (similarity: 0.842)

Generated Answer:
Emergency management includes immediate aspirin administration, 
antiplatelet therapy (P2Y12 inhibitors), anticoagulation, and 
consideration for urgent revascularization via PCI or fibrinolysis...

[Sources: Documents 12, 15, 18]
```

### Example 3: Filtered Search
```
Query: "respiratory complications"
Filter: COPD

Retrieved Sources (filtered):
✓ COPD/Chronic Bronchitis (similarity: 0.878)
✓ COPD/Emphysema (similarity: 0.865)

[Results limited to COPD category]
```

---

## 🔧 Configuration

### Model Parameters
```python
# config.py
EMBEDDING_CONFIG = {
    'model_name': 'intfloat/e5-small-v2',
    'embedding_dim': 384,
    'query_prefix': 'query: '
}

GENERATION_CONFIG = {
    'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
    'quantization': '4bit-nf4',
    'max_new_tokens': 512,
    'temperature': 0.7,
    'top_p': 0.9
}

RETRIEVAL_CONFIG = {
    'default_top_k': 5,
    'max_context_tokens': 2000,
    'similarity_threshold': 0.5
}
```

---

## 📁 Project Structure

```
clinical-rag-assistant/
├── app.py                       # Streamlit application
├── src/
│   ├── __init__.py
│   └── rag_pipeline.py         # Core RAG logic
├── data/
│   ├── README.md               # Data download guide
│   ├── chroma_db/              # Vector database
│   └── processed_documents.parquet
├── requirements.txt
├── packages.txt                # System dependencies
├── SETUP_GUIDE.md
└── README.md
```

---

## 🎓 Skills Demonstrated

### Technical Competencies
1. **RAG Architecture** - End-to-end retrieval-augmented generation
2. **Vector Databases** - ChromaDB implementation and optimization
3. **LLM Optimization** - 4-bit quantization for efficient inference
4. **Semantic Search** - E5 embeddings for clinical text
5. **Production Deployment** - Streamlit web application
6. **Healthcare NLP** - Medical text processing and domain adaptation

### Engineering Practices
- Modular code architecture
- Comprehensive documentation
- Performance benchmarking
- Resource optimization (memory, latency)
- User interface design
- Error handling and logging

---

## 🚀 Deployment

### Streamlit Cloud
1. Fork repository
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Configure secrets for API keys (if applicable)
4. Deploy from `app.py`

**Note**: GPU availability limited on free tier

### AWS EC2 (GPU Instance)
```bash
# Launch g4dn.xlarge or similar
# Install CUDA toolkit
sudo apt update && sudo apt install -y nvidia-cuda-toolkit

# Setup application
git clone https://github.com/muhammadhoud/NLP-Clinical-RAG.git
cd clinical-rag-assistant
pip install -r requirements.txt

# Run with public access
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

```bash
docker build -t clinical-rag .
docker run -p 8501:8501 --gpus all clinical-rag
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce generation tokens or switch to CPU
device = "cpu"  # In rag_pipeline.py
```

### ChromaDB Not Found
```bash
# Download from Google Drive (see data/README.md)
unzip chroma_db.zip -d data/
```

### Slow Generation
- Verify GPU availability: `torch.cuda.is_available()`
- Reduce `max_new_tokens` to 256
- Use caching for frequent queries

---

## ⚠️ Important Disclaimers

### Medical Use Limitation
This application is designed for **research and educational purposes only**. It should NOT be used for:
- Medical diagnosis
- Treatment recommendations
- Clinical decision-making
- Patient care

Always consult qualified healthcare professionals for medical advice.

### Data Privacy
The MIMIC-IV-EXT dataset contains de-identified clinical notes. Users must comply with PhysioNet's data use agreement and HIPAA regulations when handling medical data.

### Model Limitations
- Generated answers may contain factual errors
- Medical knowledge limited to training data (pre-2023)
- Cannot replace clinical expertise
- Should be verified against authoritative medical sources

---

## 🙏 Acknowledgments

- **E5 Embeddings**: Microsoft Research ([Wang et al., 2022](https://huggingface.co/intfloat/e5-small-v2))
- **Mistral-7B**: Mistral AI ([Jiang et al., 2023](https://huggingface.co/mistralai))
- **MIMIC-IV-EXT**: PhysioNet/MIT Laboratory for Computational Physiology
- **ChromaDB**: Chroma team for vector database infrastructure
- **Streamlit**: Streamlit Inc. for web framework

---

## 📄 License

MIT License - See LICENSE file for details

**Dataset License**: PhysioNet Credentialed Health Data License (separate from code license)

---

## 👤 Author

**Muhammad Houd**
- GitHub: [@muhammadhoud](https://github.com/muhammadhoud)
- LinkedIn: [Muhammad Houd](https://www.linkedin.com/in/muhammadhoud/)
- Email: 6240houd@gmail.com

---

## 📬 Contact

Questions about RAG architecture? Clinical NLP challenges? Deployment strategies?

**Open an issue** or **reach out directly** - Happy to discuss retrieval-augmented generation, healthcare AI, or production ML systems.

---

<div align="center">

**⭐ Star this repository if you found it valuable**

*"Bridging retrieval and generation: Building production-ready RAG systems for specialized domains."*

</div>
