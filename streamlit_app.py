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

# Configure page
st.set_page_config(
    page_title="Clinical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
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
            max_context_tokens=2000,
            max_input_length=4096
        )
        
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
    # Header
    st.markdown('<h1 class="main-header">🏥 Clinical RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; color: #666; margin-bottom: 2rem;'>
    AI-powered clinical decision support system using MIMIC-IV-EXT dataset<br>
    <em>E5 Embeddings + Mistral-7B (4-bit) | 934 Clinical Cases | 25 Disease Categories</em>
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
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
        
        # Query parameters
        st.subheader("Query Parameters")
        top_k = st.slider("Documents to retrieve", 1, 10, 5)
        temperature = st.slider("Generation temperature", 0.1, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max new tokens", 128, 512, 256, 32)
        
        st.divider()
        
        # Disease filter
        st.subheader("Filter by Disease")
        use_filter = st.checkbox("Enable disease filter")
        disease_category = None
        if use_filter:
            categories = [
                "Pneumonia", "Heart Failure", "Diabetes", 
                "Acute Coronary Syndrome", "Stroke", "COPD",
                "Hypertension", "Gastro-oesophageal Reflux Disease",
                "Multiple Sclerosis", "Pulmonary Embolism"
            ]
            disease_category = st.selectbox("Select disease category", categories)
        
        st.divider()
        
        # System info
        st.subheader("📊 System Info")
        if st.session_state.model_loaded:
            st.metric("Status", "🟢 Online")
            st.metric("Documents", "934")
            st.metric("Categories", "25")
        else:
            st.metric("Status", "🔴 Offline")
        
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
            
            # Dataset statistics
            st.subheader("📈 Dataset Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Cases", "511")
            col2.metric("Text Chunks", "934")
            col3.metric("Disease Categories", "25")
            col4.metric("Avg Chunk Length", "311 tokens")
        
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
    
    # Query input
    query = st.text_area(
        "🔍 Enter your clinical question:",
        height=100,
        placeholder="e.g., What are the symptoms of pneumonia with complications?"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        submit = st.button("🔎 Search & Generate Answer", type="primary", use_container_width=True)
    with col2:
        show_sources = st.checkbox("Show sources", value=True)
    with col3:
        show_metadata = st.checkbox("Show metadata", value=True)
    
    if submit and query:
        if not st.session_state.model_loaded:
            st.error("Please load models first!")
            return
        
        try:
            # Generate answer
            with st.spinner("🔍 Retrieving relevant documents..."):
                filters = {"disease_category": disease_category} if use_filter and disease_category else None
                
                start_time = time.time()
                result = st.session_state.rag_pipeline.generate_answer(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    show_progress=False
                )
                total_time = time.time() - start_time
            
            # Store in history
            st.session_state.query_history.append({
                'query': query,
                'time': total_time,
                'docs': len(result.get('sources', []))
            })
            
            # Display results
            st.success("✅ Answer generated successfully!")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            metadata = result.get('metadata', {})
            col1.metric("⏱️ Total Time", f"{total_time:.2f}s")
            col2.metric("📄 Documents", len(result.get('sources', [])))
            col3.metric("🔤 Tokens Generated", metadata.get('output_tokens', 'N/A'))
            col4.metric("🎯 Retrieval Time", f"{metadata.get('retrieval_time', 0)*1000:.0f}ms")
            
            # Answer
            st.subheader("💬 Generated Answer")
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;'>
                {result['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Sources
            if show_sources and result.get('sources'):
                st.subheader("📚 Retrieved Sources")
                
                for i, source in enumerate(result['sources'][:5], 1):
                    with st.expander(f"📄 Source {i} - {source['metadata'].get('disease_category', 'Unknown')} (Similarity: {source['similarity']:.4f})"):
                        if show_metadata:
                            st.markdown(format_metadata(source['metadata']))
                            st.divider()
                        
                        st.markdown(f"**Document Text:**")
                        st.text_area(
                            "Content",
                            source['text'][:500] + "..." if len(source['text']) > 500 else source['text'],
                            height=150,
                            key=f"source_{i}",
                            label_visibility="collapsed"
                        )
            
            # Generation metadata
            if show_metadata:
                with st.expander("🔧 Generation Metadata"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json({
                            "retrieval_time_ms": round(metadata.get('retrieval_time', 0) * 1000, 2),
                            "generation_time_ms": round(metadata.get('generation_time', 0) * 1000, 2),
                            "input_tokens": metadata.get('input_tokens', 'N/A'),
                            "output_tokens": metadata.get('output_tokens', 'N/A')
                        })
                    with col2:
                        st.json({
                            "temperature": temperature,
                            "top_k": top_k,
                            "max_new_tokens": max_tokens,
                            "disease_filter": disease_category if use_filter else "None"
                        })
        
        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")
            st.exception(e)
    
    elif submit:
        st.warning("Please enter a query first!")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>Clinical RAG Assistant</strong> | Built with Streamlit, E5, and Mistral-7B</p>
        <p><em>⚠️ For educational purposes only. Not for clinical decision-making.</em></p>
        <p><small>Models loaded from Hugging Face Hub</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
