import streamlit as st
import sys
import time
import gc
import torch
import plotly.graph_objects as go
from datetime import datetime
import json
import pandas as pd
import numpy as np

# Add path
sys.path.append('/content')

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
    page_title="Clinical RAG Assistant Ultra",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .hero-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 3rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .hero-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        transition: all 0.3s ease;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) rotate(2deg);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
    }
    
    .metric-label {
        color: rgba(255,255,255,0.9);
        font-size: 0.9rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
    }
    
    .source-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
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
    import json
    from pathlib import Path
    
    sys.path.append('/content')
    from streamlit_rag_class import RAGPipelineMistral
    
    # Clear memory before loading
    cleanup_memory()
    
    ready_flag = Path("/content/streamlit_data/pipeline_ready.flag")
    if not ready_flag.exists():
        raise FileNotFoundError("Pipeline not initialized.")
    
    config_path = Path("/content/streamlit_data/pipeline_config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load embedding model (small footprint)
    embedding_model = SentenceTransformer(config['model_name'], device=device)
    
    # Load ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=config['chroma_db_path'],
        settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_collection(name=config['collection_name'])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['generation_model_name'],
        cache_dir='/content/models'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization with memory optimization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        config['generation_model_name'],
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir='/content/models',
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16
    )
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Create pipeline with reduced defaults
    pipeline = RAGPipelineMistral(
        chroma_collection=collection,
        embedding_model=embedding_model,
        generation_model=model,
        tokenizer=tokenizer,
        max_context_tokens=1500,  # REDUCED from 2000
        max_input_length=3072      # REDUCED from 4096
    )
    
    # Override generation config for memory efficiency
    pipeline.generation_config.update({
        'max_new_tokens': 256,      # REDUCED from 512
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50,
        'repetition_penalty': 1.1,
        'do_sample': True,
    })
    
    return pipeline, config

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
    with st.spinner("🚀 Initializing AI Systems..."):
        try:
            # Clear memory first
            cleanup_memory()
            
            pipeline, config = load_rag_pipeline()
            st.session_state.rag_pipeline = pipeline
            st.session_state.pipeline_config = config
            st.session_state.initialized = True
            
            # Show initial memory
            mem = get_gpu_memory()
            if mem:
                st.toast(f"GPU Memory: {mem['allocated']:.1f}GB / {mem['total']:.1f}GB", icon="🎮")
            
            st.rerun()
        except FileNotFoundError:
            st.error("❌ Pipeline not found. Run notebook cells first.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")
            st.exception(e)
            st.stop()

# ============================================================================
# HERO HEADER
# ============================================================================
st.markdown("""
<div class="hero-header">
    <h1>✨ Clinical RAG Assistant Ultra</h1>
    <p style="color: white; font-size: 1.2rem;">
        Memory-optimized for T4 GPU • Advanced RAG System
    </p>
    <div style="margin-top: 1rem;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.5rem; color: white;">⚡ Fast</span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.5rem; color: white;">🧠 Smart</span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.5rem; color: white;">💾 Optimized</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - OPTIMIZED SETTINGS
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Top K (reduced default)
    top_k = st.slider("📊 Top K Results", 1, 5, 3,  # MAX 5, DEFAULT 3
                     help="Fewer results = less memory")
    
    # Temperature
    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.7, 0.1)
    
    # Max tokens (NEW - user control)
    max_tokens = st.slider("📝 Max Output Tokens", 128, 512, 256, 64,
                          help="Lower = less memory, faster generation")
    
    # Filter toggle
    use_filter = st.checkbox("🔍 Enable Disease Filter", value=False)
    
    selected_category = None
    if use_filter:
        categories = [
            "Pneumonia", "Diabetes", "Heart Failure", "Stroke", 
            "COPD", "Hypertension", "Acute Coronary Syndrome"
        ]
        selected_category = st.selectbox("Select Category", options=categories)
    
    st.markdown("---")
    
    # GPU Memory Monitor
    st.markdown("### 🎮 GPU Status")
    mem = get_gpu_memory()
    if mem:
        usage_pct = (mem['allocated'] / mem['total']) * 100
        st.metric("Memory Used", f"{mem['allocated']:.1f}GB", 
                 delta=f"{usage_pct:.0f}%")
        
        # Memory warning
        if usage_pct > 80:
            st.warning("⚠️ High memory usage", icon="⚠️")
        elif usage_pct > 60:
            st.info("💡 Moderate usage", icon="💡")
        else:
            st.success("✅ Good memory", icon="✅")
    
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    st.metric("Total Queries", st.session_state.total_queries)
    if st.session_state.avg_response_time > 0:
        st.metric("Avg Time", f"{st.session_state.avg_response_time:.1f}s")
    
    # Manual cleanup button
    if st.button("🧹 Clear GPU Memory", use_container_width=True):
        cleanup_memory()
        st.toast("✅ Memory cleared!", icon="✅")
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
        height=120,
        placeholder="e.g., What are the symptoms of pneumonia?",
        help="Keep queries focused for better results"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎲 Random", use_container_width=True):
        query = np.random.choice(example_queries)
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PROCESS QUERY - MEMORY OPTIMIZED
# ============================================================================
if search_clicked and query.strip():
    # Clear memory before processing
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
        
        # Generate with memory-optimized parameters
        result = st.session_state.rag_pipeline.generate_answer(
            query=query,
            top_k=top_k,              # User-controlled (reduced)
            filters=filters,
            temperature=temperature,
            max_new_tokens=max_tokens, # User-controlled
            show_progress=False
        )
        
        progress_bar.progress(80)
        
        # Aggressive cleanup after generation
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
        
        # ====================================================================
        # RESULTS DISPLAY
        # ====================================================================
        
        st.markdown("### 📊 Query Performance")
        
        metric_cols = st.columns(5)
        
        retrieval_time = result['metadata'].get('retrieval_time', 0) * 1000
        generation_time = result['metadata'].get('generation_time', 0)
        output_tokens = result['metadata'].get('output_tokens', 
                                               result['metadata'].get('total_tokens', 0))
        
        metrics_data = [
            ("⚡", len(result.get('sources', [])), "Sources"),
            ("🎯", f"{retrieval_time:.0f}ms", "Retrieval"),
            ("🤖", f"{generation_time:.1f}s", "Generation"),
            ("📝", output_tokens, "Tokens"),
            ("⏱️", f"{total_time:.2f}s", "Total")
        ]
        
        for col, (icon, value, label) in zip(metric_cols, metrics_data):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # AI Answer
        st.markdown("### 🤖 AI-Generated Answer")
        
        answer_text = result.get('answer', 'No answer generated')
        st.info(answer_text)
        
        # Copy button
        if st.button("📋 Copy Answer", use_container_width=False):
            st.toast("✅ Copied!", icon="✅")
        
        st.markdown("---")
        
        # Sources (collapsed by default to save screen space)
        sources = result.get('sources', [])
        if sources:
            st.markdown("### 📚 Retrieved Sources")
            
            similarities = [s.get('similarity', 0) for s in sources]
            avg_similarity = np.mean(similarities) if similarities else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Confidence", f"{avg_similarity:.1%}")
            with col2:
                st.metric("📄 Sources", len(sources))
            with col3:
                high_q = sum(1 for s in similarities if s > 0.7)
                st.metric("⭐ High Quality", high_q)
            
            for i, source in enumerate(sources, 1):
                similarity = source.get('similarity', 0)
                category = source.get('metadata', {}).get('disease_category', 'Unknown')
                
                with st.expander(f"📄 #{i}: {category} ({similarity:.0%})", 
                               expanded=(i == 1)):
                    st.markdown(f"**Confidence:** {similarity:.1%}")
                    
                    # Confidence bar
                    st.progress(similarity)
                    
                    text_content = source.get('text', '')[:500]  # Limit preview
                    st.text_area("Preview", value=text_content, height=150, 
                               key=f"src_{i}", label_visibility="collapsed")
        
        st.success(f"✅ Completed in {total_time:.1f}s", icon="✅")
        
        # Show final memory state
        mem = get_gpu_memory()
        if mem:
            st.caption(f"🎮 GPU: {mem['allocated']:.1f}GB / {mem['total']:.1f}GB")
        
    except RuntimeError as e:
        progress_bar.empty()
        status_text.empty()
        
        if "out of memory" in str(e).lower():
            st.error("❌ GPU Out of Memory!", icon="❌")
            st.warning("""
            **Quick Fixes:**
            1. Click "🧹 Clear GPU Memory" in sidebar
            2. Reduce "Max Output Tokens" to 128
            3. Reduce "Top K Results" to 1-2
            4. Restart the notebook runtime if issue persists
            """)
            
            # Attempt recovery
            cleanup_memory()
        else:
            st.error(f"❌ Error: {str(e)}", icon="❌")
            
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error: {str(e)}", icon="❌")
        
elif search_clicked:
    st.warning("⚠️ Please enter a question", icon="⚠️")

# ============================================================================
# QUERY HISTORY
# ============================================================================
if st.session_state.query_history:
    st.markdown("---")
    with st.expander("📜 Query History", expanded=False):
        for i, item in enumerate(st.session_state.query_history[:5]):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"{i+1}. {item['query'][:60]}...")
            with col2:
                st.caption(item['timestamp'])

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: white;">
    <p style="color: rgba(255,255,255,0.8);">
        🏥 Clinical RAG Assistant • Memory-Optimized Edition
    </p>
    <p style="font-size: 0.8rem; color: rgba(255,255,255,0.6);">
        ⚠️ Research purposes only • Optimized for Colab T4 GPU
    </p>
</div>
""", unsafe_allow_html=True)
