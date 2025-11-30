"""
Clinical RAG System - Streamlit Interface
===========================================
Streamlit app for clinical question answering using E5 + Mistral-7B RAG pipeline.
Deployed on Streamlit Cloud with ChromaDB and pre-computed embeddings.
"""

import streamlit as st
import sys
from pathlib import Path
import time
import json
import gc
from datetime import datetime
import os
import numpy as np

# Configure page
st.set_page_config(
    page_title="Clinical RAG Assistant Ultra",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS - Enhanced UI from memory-optimized version
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
    
    .model-loading {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        background-size: 400% 400%;
        animation: gradient 3s ease infinite;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'loading_progress' not in st.session_state:
    st.session_state.loading_progress = 0
if 'loading_status' not in st.session_state:
    st.session_state.loading_status = "Ready to load models"

# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================
def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass

def get_gpu_memory():
    """Get current GPU memory stats"""
    try:
        import torch
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
    except:
        pass
    return None

@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    """Load RAG pipeline components with Hugging Face Hub integration."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    
    try:
        # Import RAG pipeline class
        sys.path.append(str(Path(__file__).parent / "src"))
        from rag_pipeline import RAGPipelineMistral
        
        # Device configuration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.session_state.loading_status = f"🖥️ Using device: {device}"
        
        # Load E5 embedding model
        st.session_state.loading_status = "📥 Loading E5 embedding model..."
        embedding_model = SentenceTransformer(
            'intfloat/e5-small-v2',
            device=device
        )
        st.session_state.loading_progress = 25
        
        # Load ChromaDB
        st.session_state.loading_status = "📚 Loading ChromaDB..."
        chroma_client = chromadb.PersistentClient(
            path="./data/chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        collection = chroma_client.get_collection(name="clinical_notes")
        st.session_state.loading_progress = 40
        
        # Configure 4-bit quantization for Mistral
        st.session_state.loading_status = "⚙️ Configuring 4-bit quantization..."
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load Mistral tokenizer from Hugging Face Hub
        st.session_state.loading_status = "🔤 Loading Mistral tokenizer..."
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        st.session_state.loading_progress = 60
        
        # Load Mistral model (4-bit quantized) from Hugging Face Hub
        st.session_state.loading_status = "🧠 Loading Mistral-7B model (this may take a few minutes)..."
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        st.session_state.loading_progress = 85
        
        # Initialize RAG pipeline
        st.session_state.loading_status = "🔧 Creating RAG pipeline..."
        rag_pipeline = RAGPipelineMistral(
            chroma_collection=collection,
            embedding_model=embedding_model,
            generation_model=model,
            tokenizer=tokenizer,
            max_context_tokens=1500,  # Reduced for memory efficiency
            max_input_length=3072     # Reduced for memory efficiency
        )
        
        # Override generation config for memory efficiency
        rag_pipeline.generation_config.update({
            'max_new_tokens': 256,    # Reduced from 512
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
            'do_sample': True,
        })
        
        st.session_state.loading_progress = 100
        st.session_state.loading_status = "✅ Models loaded successfully!"
        
        return rag_pipeline, device
        
    except Exception as e:
        st.session_state.loading_status = f"❌ Error: {str(e)}"
        return None, None

def load_models_with_progress():
    """Load models with progress tracking."""
    import threading
    
    def load_thread():
        pipeline, device = load_rag_pipeline()
        if pipeline:
            st.session_state.rag_pipeline = pipeline
            st.session_state.model_loaded = True
    
    # Start loading in a separate thread
    thread = threading.Thread(target=load_thread)
    thread.daemon = True
    thread.start()

def format_metadata(metadata):
    """Format metadata for display."""
    display_fields = {
        'disease_category': 'Disease Category',
        'disease_subtype': 'Disease Subtype',
        'chunk_index': 'Chunk',
        'total_chunks': 'Total Chunks'
    }
    
    formatted = []
    for key, label in display_fields.items():
        if key in metadata:
            value = metadata[key]
            if key == 'chunk_index':
                formatted.append(f"**{label}**: {value + 1}/{metadata.get('total_chunks', '?')}")
            else:
                formatted.append(f"**{label}**: {value}")
    
    return " | ".join(formatted)

def check_disk_space():
    """Check if we have enough disk space for model download."""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        return free_gb >= 8  # Need at least 8GB free
    except:
        return True  # If we can't check, assume it's fine

def main():
    # Enhanced Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1>✨ Clinical RAG Assistant Ultra</h1>
        <p style="color: white; font-size: 1.2rem;">
            Memory-optimized • Advanced RAG System • Hugging Face Hub Integration
        </p>
        <div style="margin-top: 1rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.5rem; color: white;">⚡ Fast</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.5rem; color: white;">🧠 Smart</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.5rem; color: white;">💾 Optimized</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model loading section
        if not st.session_state.model_loaded:
            st.markdown("### 🚀 Model Loading")
            
            # Disk space check
            if not check_disk_space():
                st.warning("⚠️ Low disk space detected. Model download may fail.")
            
            st.info("""
            **Model Download Info:**
            - Mistral-7B: ~4GB (4-bit quantized)
            - E5-small: ~130MB
            - First load: 5-10 minutes
            - Subsequent loads: Instant (cached)
            """)
            
            if st.button("🚀 Download & Load Models", type="primary", use_container_width=True):
                st.session_state.loading_progress = 0
                st.session_state.loading_status = "Starting model download..."
                load_models_with_progress()
                
        else:
            st.success("✅ Models loaded and ready")
            
            # Model info
            st.markdown("### 📊 Model Information")
            st.metric("Mistral-7B", "4-bit Quantized")
            st.metric("E5 Embeddings", "Small-v2")
            st.metric("Status", "🟢 Online")
        
        # Loading progress display
        if st.session_state.loading_progress > 0 and not st.session_state.model_loaded:
            st.markdown("### 📥 Download Progress")
            st.progress(st.session_state.loading_progress / 100)
            st.caption(st.session_state.loading_status)
            
            if st.session_state.loading_progress == 100:
                time.sleep(1)  # Let user see the completion
                st.rerun()
        
        st.divider()
        
        # Enhanced Query Parameters
        st.subheader("Query Parameters")
        top_k = st.slider("📊 Top K Results", 1, 5, 3, help="Fewer results = less memory")
        temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("📝 Max Output Tokens", 128, 512, 256, 64, help="Lower = less memory, faster generation")
        
        st.divider()
        
        # Disease filter
        st.subheader("Filter by Disease")
        use_filter = st.checkbox("🔍 Enable Disease Filter", value=False)
        disease_category = None
        if use_filter:
            categories = [
                "Pneumonia", "Heart Failure", "Diabetes", 
                "Acute Coronary Syndrome", "Stroke", "COPD",
                "Hypertension", "Gastro-oesophageal Reflux Disease",
                "Multiple Sclerosis", "Pulmonary Embolism"
            ]
            disease_category = st.selectbox("Select Category", options=categories)
        
        st.divider()
        
        # GPU Memory Monitor (Enhanced)
        st.markdown("### 🎮 System Status")
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
        if st.button("🧹 Clear Memory", use_container_width=True):
            cleanup_memory()
            st.toast("✅ Memory cleared!", icon="✅")
            time.sleep(0.5)
            st.rerun()
        
        # Query history
        st.divider()
        st.subheader("📜 Query History")
        if st.session_state.query_history:
            for i, hist in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Query {len(st.session_state.query_history) - i}"):
                    st.text(hist['query'][:100] + "...")
                    st.caption(f"⏱️ {hist['time']:.2f}s | 🎯 {hist['docs']} docs")
        else:
            st.info("No queries yet")
    
    # Main content
    if not st.session_state.model_loaded:
        if st.session_state.loading_progress == 0:
            st.info("👈 Click **Download & Load Models** in the sidebar to start")
            
            # Show example queries
            st.subheader("Example Queries")
            examples = [
                "What are the common symptoms of pneumonia?",
                "How is acute coronary syndrome diagnosed and managed?",
                "What complications should be monitored in diabetic patients?",
                "What are the warning signs of pulmonary embolism?",
                "Describe the diagnostic approach for suspected sepsis."
            ]
            
            cols = st.columns(2)
            for i, example in enumerate(examples):
                with cols[i % 2]:
                    st.info(f"💡 {example}")
            
            # Dataset statistics with enhanced cards
            st.subheader("📈 Dataset Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">511</div>
                    <div class="metric-label">Total Cases</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">934</div>
                    <div class="metric-label">Text Chunks</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">25</div>
                    <div class="metric-label">Categories</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">311</div>
                    <div class="metric-label">Avg Tokens</div>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            # Show loading animation
            st.markdown(f"""
            <div class="model-loading">
                <h3>🚀 Loading AI Models</h3>
                <p>{st.session_state.loading_status}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            st.progress(st.session_state.loading_progress / 100)
            
            # Loading tips
            with st.expander("💡 Loading Tips"):
                st.markdown("""
                - **First time?** This may take 5-10 minutes
                - **Slow internet?** Be patient, models are downloading
                - **Stuck?** Refresh and try again
                - **Memory issues?** Models are 4-bit quantized for efficiency
                """)
        
        return
    
    # Enhanced Query Interface
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

    # Enhanced Query Processing
    if search_clicked and query.strip():
        # Clear memory before processing
        cleanup_memory()
        
        st.session_state.query_history.insert(0, {
            'query': query,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        st.session_state.total_queries += 1
        
        filters = {"disease_category": disease_category} if use_filter and disease_category else None
        
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
            
            # Enhanced Results Display
            st.markdown("### 📊 Query Performance")
            
            metric_cols = st.columns(5)
            metadata = result.get('metadata', {})
            retrieval_time = metadata.get('retrieval_time', 0) * 1000
            generation_time = metadata.get('generation_time', 0)
            output_tokens = metadata.get('output_tokens', 'N/A')
            sources_count = len(result.get('sources', []))
            
            metrics_data = [
                ("⚡", sources_count, "Sources"),
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
            
            # Enhanced Answer Display
            st.markdown("### 🤖 AI-Generated Answer")
            answer_text = result.get('answer', 'No answer generated')
            st.info(answer_text)
            
            # Copy button
            if st.button("📋 Copy Answer", use_container_width=False):
                st.toast("✅ Copied!", icon="✅")
            
            st.markdown("---")
            
            # Enhanced Sources Display
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
                        st.progress(similarity)
                        
                        text_content = source.get('text', '')[:500]
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
                1. Click "🧹 Clear Memory" in sidebar
                2. Reduce "Max Output Tokens" to 128
                3. Reduce "Top K Results" to 1-2
                4. Restart the app if issue persists
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

    # Enhanced Query History
    if st.session_state.query_history:
        st.markdown("---")
        with st.expander("📜 Query History", expanded=False):
            for i, item in enumerate(st.session_state.query_history[:5]):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"{i+1}. {item['query'][:60]}...")
                with col2:
                    st.caption(item['timestamp'])

    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: white;">
        <p style="color: rgba(255,255,255,0.8);">
            🏥 Clinical RAG Assistant Ultra • Memory-Optimized Edition
        </p>
        <p style="font-size: 0.8rem; color: rgba(255,255,255,0.6);">
            ⚠️ Research purposes only • Hugging Face Hub Integration
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
