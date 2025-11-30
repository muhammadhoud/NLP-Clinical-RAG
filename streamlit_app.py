import streamlit as st
import sys
import os
import time
import gc
import torch
from datetime import datetime
import json
import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================
def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_gpu_memory():
    """Get current GPU memory stats"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': total - reserved,
            'total': total
        }
    return None

# ============================================================================
# CHROMADB MANAGEMENT
# ============================================================================
def initialize_chromadb_collection():
    """Initialize ChromaDB collection with sample clinical data"""
    try:
        chroma_path = "data/chroma_db"
        
        # Create directory if it doesn't exist
        os.makedirs(chroma_path, exist_ok=True)
        
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Check if collection exists
        collections = chroma_client.list_collections()
        collection_names = [col.name for col in collections]
        
        if "clinical_notes" in collection_names:
            collection = chroma_client.get_collection("clinical_notes")
            st.success(f"✅ Collection 'clinical_notes' loaded with {collection.count()} documents")
            return collection
        
        # Create new collection
        st.info("🔄 Creating new 'clinical_notes' collection...")
        collection = chroma_client.create_collection(
            name="clinical_notes",
            metadata={"description": "Clinical notes for RAG system"}
        )
        
        # Add sample clinical notes
        sample_documents = [
            "Patient presents with fever, cough, and shortness of breath. Chest X-ray shows consolidation in right lower lobe. Diagnosis: Community-acquired pneumonia.",
            "Hypertension management: Patient's blood pressure is 145/92 mmHg. Current medications include lisinopril 10mg daily. Lifestyle modifications discussed.",
            "Diabetes follow-up: HbA1c is 7.2%. Patient reports adherence to metformin 500mg twice daily. Foot examination shows no neuropathy signs.",
            "Heart failure exacerbation: Patient presents with dyspnea on exertion and bilateral lower extremity edema. Echocardiogram shows reduced ejection fraction of 35%.",
            "Stroke evaluation: CT head shows acute ischemic changes in left MCA territory. Patient has right-sided weakness and aphasia. NIH stroke scale: 12.",
            "COPD management: Patient with chronic bronchitis presents with increased sputum production. Spirometry shows FEV1/FVC ratio of 0.58. On bronchodilator therapy.",
            "Acute coronary syndrome: Patient with chest pain radiating to left arm. ECG shows ST-segment elevation in anterior leads. Troponin elevated at 2.4 ng/mL.",
            "Asthma exacerbation: Patient presents with wheezing and respiratory distress. Peak flow 45% of personal best. Started on nebulized albuterol and corticosteroids.",
            "Renal function: Patient with chronic kidney disease stage 3. Creatinine 1.8 mg/dL, eGFR 45 mL/min/1.73m². Monitoring for proteinuria.",
            "Mental health: Patient reports depressive symptoms including anhedonia and sleep disturbance. PHQ-9 score: 15. Starting SSRI therapy."
        ]
        
        sample_metadata = [
            {"disease_category": "Pneumonia", "document_type": "clinical_note", "severity": "moderate"},
            {"disease_category": "Hypertension", "document_type": "follow_up", "severity": "mild"},
            {"disease_category": "Diabetes", "document_type": "follow_up", "severity": "moderate"},
            {"disease_category": "Heart Failure", "document_type": "acute_care", "severity": "severe"},
            {"disease_category": "Stroke", "document_type": "emergency", "severity": "severe"},
            {"disease_category": "COPD", "document_type": "chronic_care", "severity": "moderate"},
            {"disease_category": "Acute Coronary Syndrome", "document_type": "emergency", "severity": "severe"},
            {"disease_category": "Asthma", "document_type": "acute_care", "severity": "moderate"},
            {"disease_category": "Renal Disease", "document_type": "chronic_care", "severity": "moderate"},
            {"disease_category": "Mental Health", "document_type": "evaluation", "severity": "moderate"}
        ]
        
        # Add documents to collection
        for i, (doc, meta) in enumerate(zip(sample_documents, sample_metadata)):
            collection.add(
                documents=[doc],
                metadatas=[meta],
                ids=[f"clinical_note_{i+1}"]
            )
        
        st.success(f"✅ Created 'clinical_notes' collection with {len(sample_documents)} sample documents")
        return collection
        
    except Exception as e:
        st.error(f"❌ Failed to initialize ChromaDB: {e}")
        return None

def get_chromadb_collection():
    """Get or create ChromaDB collection"""
    try:
        chroma_path = "data/chroma_db"
        
        # Check if ChromaDB directory exists
        if not os.path.exists(chroma_path):
            st.warning("📁 ChromaDB directory not found. Creating new database...")
            return initialize_chromadb_collection()
        
        # Initialize client
        chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # List collections
        collections = chroma_client.list_collections()
        
        if not collections:
            st.warning("📊 No collections found. Creating new collection...")
            return initialize_chromadb_collection()
        
        # Try to get clinical_notes collection
        try:
            collection = chroma_client.get_collection("clinical_notes")
            st.success(f"✅ Loaded 'clinical_notes' collection with {collection.count()} documents")
            return collection
        except Exception as e:
            st.warning(f"⚠️ Collection 'clinical_notes' not found: {e}")
            st.info("Available collections:")
            for col in collections:
                st.write(f"- {col.name} ({col.count()} documents)")
            
            # Offer to create collection
            if st.button("🔄 Create Clinical Notes Collection"):
                return initialize_chromadb_collection()
            return None
            
    except Exception as e:
        st.error(f"❌ ChromaDB error: {e}")
        return None

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Clinical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .hero-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .hero-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: white;
    }
    
    .metric-label {
        color: rgba(255,255,255,0.9);
        font-size: 0.8rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LAZY LOADING FUNCTION
# ============================================================================
@st.cache_resource
def load_rag_pipeline():
    """Load RAG pipeline components with memory optimization"""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    # Import from src directory
    try:
        from src.rag_pipeline import RAGPipelineMistral
    except ImportError:
        st.error("❌ Could not import RAGPipelineMistral from src directory")
        return None, None
    
    # Clear memory before loading
    cleanup_memory()
    
    # Get ChromaDB collection
    collection = get_chromadb_collection()
    if collection is None:
        return None, None
    
    # Configuration
    config = {
        'chroma_db_path': "data/chroma_db",
        'collection_name': "clinical_notes",
        'model_name': "intfloat/e5-small-v2",
        'generation_model_name': "mistralai/Mistral-7B-Instruct-v0.2"
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"🖥️ Using device: {device}")

    try:
        # Load embedding model
        st.info("📥 Loading embedding model...")
        embedding_model = SentenceTransformer(config['model_name'], device=device)
        
        # Load tokenizer
        st.info("🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config['generation_model_name'])
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure 4-bit quantization
        st.info("⚙️ Configuring model quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load model
        st.info("🧠 Loading language model...")
        model = AutoModelForCausalLM.from_pretrained(
            config['generation_model_name'],
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Create pipeline
        st.info("🔧 Creating RAG pipeline...")
        pipeline = RAGPipelineMistral(
            chroma_collection=collection,
            embedding_model=embedding_model,
            generation_model=model,
            tokenizer=tokenizer,
            max_context_tokens=1500,
            max_input_length=3072
        )
        
        # Update generation config
        pipeline.generation_config.update({
            'max_new_tokens': 256,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
            'do_sample': True,
        })
        
        st.success("✅ RAG pipeline initialized successfully!")
        return pipeline, config
        
    except Exception as e:
        st.error(f"❌ Error loading pipeline: {str(e)}")
        return None, None

# ============================================================================
# SESSION STATE
# ============================================================================
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
    st.session_state.initialized = False
    st.session_state.query_history = []
    st.session_state.total_queries = 0
    st.session_state.avg_response_time = 0

# ============================================================================
# INITIALIZE PIPELINE
# ============================================================================
if not st.session_state.initialized:
    with st.spinner("🚀 Initializing Clinical RAG System..."):
        try:
            # Clear memory first
            cleanup_memory()
            
            pipeline, config = load_rag_pipeline()
            if pipeline is None:
                st.error("❌ Failed to initialize pipeline.")
                st.markdown("""
                <div class="warning-box">
                    <h4>💡 Troubleshooting Tips:</h4>
                    <ul>
                        <li>Make sure the ChromaDB data directory exists at <code>data/chroma_db/</code></li>
                        <li>Check if all required models are available</li>
                        <li>Verify your internet connection for model downloads</li>
                        <li>Ensure sufficient GPU memory is available</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                st.stop()
                
            st.session_state.rag_pipeline = pipeline
            st.session_state.pipeline_config = config
            st.session_state.initialized = True
            
        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")
            st.stop()

# ============================================================================
# HERO HEADER
# ============================================================================
st.markdown("""
<div class="hero-header">
    <h1>🏥 Clinical RAG Assistant</h1>
    <p style="color: white; font-size: 1.2rem;">
        AI-powered clinical document retrieval and question answering
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    top_k = st.slider("📊 Top K Results", 1, 5, 3)
    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("📝 Max Output Tokens", 128, 512, 256, 64)
    
    use_filter = st.checkbox("🔍 Enable Disease Filter", value=False)
    selected_category = None
    if use_filter:
        categories = [
            "Pneumonia", "Diabetes", "Heart Failure", "Stroke", 
            "COPD", "Hypertension", "Acute Coronary Syndrome", "Asthma",
            "Renal Disease", "Mental Health"
        ]
        selected_category = st.selectbox("Select Category", options=categories)
    
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    st.metric("Total Queries", st.session_state.total_queries)
    if st.session_state.avg_response_time > 0:
        st.metric("Avg Time", f"{st.session_state.avg_response_time:.1f}s")
    
    # Database management
    st.markdown("---")
    st.markdown("### 🗄️ Database")
    
    if st.session_state.rag_pipeline:
        collection = st.session_state.rag_pipeline.chroma_collection
        if collection:
            st.metric("Documents", collection.count())
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh DB", use_container_width=True):
            st.session_state.initialized = False
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Memory", use_container_width=True):
            cleanup_memory()
            st.toast("✅ Memory cleared!")
            time.sleep(0.5)

# ============================================================================
# MAIN QUERY INTERFACE
# ============================================================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("## 💬 Ask Your Clinical Question")

example_queries = [
    "What are the main symptoms of pneumonia?",
    "How is diabetes diagnosed and managed?",
    "What are the treatment options for heart failure?",
    "What are the risk factors for stroke?",
    "How is hypertension managed?",
    "What are the symptoms of COPD?",
]

col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., What are the symptoms and treatment for pneumonia?",
        help="Ask about symptoms, diagnoses, treatments, risk factors, etc."
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎲 Example", use_container_width=True):
        query = np.random.choice(example_queries)
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PROCESS QUERY
# ============================================================================
if search_clicked and query.strip():
    cleanup_memory()
    
    st.session_state.query_history.insert(0, {
        'query': query,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    st.session_state.total_queries += 1
    
    filters = {"disease_category": selected_category} if use_filter and selected_category else None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.info("🔍 Analyzing query...")
    progress_bar.progress(20)
    
    try:
        start_time = time.time()
        
        status_text.info("🧠 Retrieving documents...")
        progress_bar.progress(40)
        
        # Generate answer
        result = st.session_state.rag_pipeline.generate_answer(
            query=query,
            top_k=top_k,
            filters=filters,
            temperature=temperature,
            max_new_tokens=max_tokens,
            show_progress=False
        )
        
        progress_bar.progress(80)
        cleanup_memory()
        
        total_time = time.time() - start_time
        st.session_state.avg_response_time = (
            (st.session_state.avg_response_time * (st.session_state.total_queries - 1) + total_time) 
            / st.session_state.total_queries
        )
        
        progress_bar.progress(100)
        time.sleep(0.3)
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.markdown("### 📊 Results")
        
        metric_cols = st.columns(4)
        sources_count = len(result.get('sources', []))
        retrieval_time = result['metadata'].get('retrieval_time', 0) * 1000
        generation_time = result['metadata'].get('generation_time', 0)
        output_tokens = result['metadata'].get('output_tokens', 0)
        
        metrics_data = [
            ("⚡", sources_count, "Sources"),
            ("🎯", f"{retrieval_time:.0f}ms", "Retrieval"),
            ("🤖", f"{generation_time:.1f}s", "Generation"),
            ("📝", output_tokens, "Tokens")
        ]
        
        for col, (icon, value, label) in zip(metric_cols, metrics_data):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # AI Answer
        st.markdown("### 🤖 AI Answer")
        answer_text = result.get('answer', 'No answer generated.')
        st.info(answer_text)
        
        # Sources
        sources = result.get('sources', [])
        if sources:
            st.markdown("### 📚 Retrieved Sources")
            
            for i, source in enumerate(sources, 1):
                similarity = source.get('similarity', 0)
                metadata = source.get('metadata', {})
                category = metadata.get('disease_category', 'Unknown')
                doc_type = metadata.get('document_type', 'Clinical Note')
                
                with st.expander(f"📄 #{i}: {category} - {doc_type} ({similarity:.0%} match)"):
                    st.markdown(f"**Confidence:** {similarity:.1%}")
                    st.progress(similarity)
                    
                    text_content = source.get('text', '')[:500] + "..." if len(source.get('text', '')) > 500 else source.get('text', '')
                    st.text_area("Content", value=text_content, height=120, 
                               key=f"src_{i}", label_visibility="collapsed")
        
        st.success(f"✅ Analysis completed in {total_time:.1f}s")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error processing query: {str(e)}")
        
elif search_clicked:
    st.warning("⚠️ Please enter a question")

# ============================================================================
# QUERY HISTORY
# ============================================================================
if st.session_state.query_history:
    st.markdown("---")
    with st.expander("📜 Recent Queries"):
        for i, item in enumerate(st.session_state.query_history[:5]):
            st.caption(f"{i+1}. {item['query'][:80]}... ({item['timestamp']})")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: white;">
    <p style="color: rgba(255,255,255,0.8);">
        🏥 Clinical RAG Assistant • Medical Document Retrieval System
    </p>
    <p style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">
        Note: This system uses sample clinical data for demonstration purposes.
    </p>
</div>
""", unsafe_allow_html=True)
