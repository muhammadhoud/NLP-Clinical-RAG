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
    import chromadb
    from chromadb.config import Settings
    
    # Import from src directory
    from src.rag_pipeline import RAGPipelineMistral
    
    # Clear memory before loading
    cleanup_memory()
    
    # Configuration
    config = {
        'chroma_db_path': "data/chroma_db",
        'collection_name': "clinical_notes",
        'model_name': "intfloat/e5-small-v2",
        'generation_model_name': "mistralai/Mistral-7B-Instruct-v0.2"
    }
    
    # Check if ChromaDB exists
    if not os.path.exists(config['chroma_db_path']):
        raise FileNotFoundError(f"ChromaDB not found at {config['chroma_db_path']}. Please download the data.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load embedding model
        embedding_model = SentenceTransformer(config['model_name'], device=device)
        
        # Load ChromaDB
        chroma_client = chromadb.PersistentClient(
            path=config['chroma_db_path'],
            settings=Settings(anonymized_telemetry=False)
        )
        collection = chroma_client.get_collection(name=config['collection_name'])
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['generation_model_name'])
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config['generation_model_name'],
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )
        
        # Create pipeline
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
        
        return pipeline, config
        
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        raise

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
            st.session_state.rag_pipeline = pipeline
            st.session_state.pipeline_config = config
            st.session_state.initialized = True
            
            st.success("✅ Pipeline initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")
            st.info("💡 Make sure you've downloaded the ChromaDB data to the data/ directory")
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
            "COPD", "Hypertension", "Acute Coronary Syndrome"
        ]
        selected_category = st.selectbox("Select Category", options=categories)
    
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    st.metric("Total Queries", st.session_state.total_queries)
    if st.session_state.avg_response_time > 0:
        st.metric("Avg Time", f"{st.session_state.avg_response_time:.1f}s")
    
    if st.button("🧹 Clear Memory", use_container_width=True):
        cleanup_memory()
        st.toast("✅ Memory cleared!")
        time.sleep(0.5)
        st.rerun()

# ============================================================================
# MAIN QUERY INTERFACE
# ============================================================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("## 💬 Ask Your Clinical Question")

example_queries = [
    "What are the main symptoms of pneumonia?",
    "How is diabetes diagnosed?",
    "What causes heart failure?",
    "List stroke risk factors",
]

col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., What are the symptoms of pneumonia?",
        help="Ask about symptoms, diagnoses, treatments, etc."
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
        retrieval_time = result['metadata'].get('retrieval_time', 0) * 1000
        generation_time = result['metadata'].get('generation_time', 0)
        output_tokens = result['metadata'].get('output_tokens', 0)
        
        metrics_data = [
            ("⚡", len(result.get('sources', [])), "Sources"),
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
                category = source.get('metadata', {}).get('disease_category', 'Unknown')
                
                with st.expander(f"📄 #{i}: {category} ({similarity:.0%})"):
                    st.markdown(f"**Confidence:** {similarity:.1%}")
                    st.progress(similarity)
                    
                    text_content = source.get('text', '')[:400]
                    st.text_area("Preview", value=text_content, height=120, 
                               key=f"src_{i}", label_visibility="collapsed")
        
        st.success(f"✅ Completed in {total_time:.1f}s")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error: {str(e)}")
        
elif search_clicked:
    st.warning("⚠️ Please enter a question")

# ============================================================================
# QUERY HISTORY
# ============================================================================
if st.session_state.query_history:
    st.markdown("---")
    with st.expander("📜 Recent Queries"):
        for i, item in enumerate(st.session_state.query_history[:3]):
            st.caption(f"{i+1}. {item['query'][:80]}... ({item['timestamp']})")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: white;">
    <p style="color: rgba(255,255,255,0.8);">
        🏥 Clinical RAG Assistant • Medical Document Retrieval System
    </p>
</div>
""", unsafe_allow_html=True)
