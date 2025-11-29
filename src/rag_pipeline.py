"""
RAG Pipeline Core Implementation
=================================
Production RAG pipeline combining ChromaDB retrieval with Mistral-7B generation.
Optimized for Streamlit deployment with memory-efficient 4-bit quantization.

Author: Clinical RAG Assistant
Version: 1.0
"""

import time
import torch
import numpy as np
from typing import List, Dict, Optional
import gc


class RAGPipelineMistral:
    """
    Production-ready RAG pipeline combining ChromaDB retrieval 
    with Mistral-7B-Instruct generation.
    
    Features:
    - E5-small-v2 embeddings with query prefix
    - ChromaDB semantic search
    - Mistral-7B (4-bit quantized) for generation
    - Memory-efficient with automatic cleanup
    - Clinical domain-specific prompting
    """
    
    def __init__(self, 
                 chroma_collection,
                 embedding_model,
                 generation_model,
                 tokenizer,
                 max_context_tokens=2000,
                 max_input_length=4096):
        """
        Initialize RAG pipeline.
        
        Args:
            chroma_collection: ChromaDB collection instance
            embedding_model: SentenceTransformer (E5-small-v2)
            generation_model: Mistral-7B model (4-bit quantized)
            tokenizer: Mistral tokenizer
            max_context_tokens: Maximum tokens for retrieved context
            max_input_length: Maximum input length for model
        """
        self.collection = chroma_collection
        self.embedding_model = embedding_model
        self.model = generation_model
        self.tokenizer = tokenizer
        
        # Configuration
        self.max_context_tokens = max_context_tokens
        self.max_input_length = max_input_length
        self.query_prefix = "query: "  # E5 requires this prefix
        
        # Default generation parameters
        self.default_generation_config = {
            'max_new_tokens': 256,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
            'do_sample': True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        print("✓ RAG Pipeline initialized successfully")
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query with E5 model using proper "query: " prefix.
        
        Args:
            query: User query string
            
        Returns:
            Normalized query embedding (384-dim)
        """
        # CRITICAL: E5 requires "query: " prefix for queries
        prefixed_query = f"{self.query_prefix}{query}"
        
        with torch.no_grad():
            embedding = self.embedding_model.encode(
                prefixed_query,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
                convert_to_numpy=True
            )
        
        return embedding
    
    def retrieve_documents(self, 
                          query: str, 
                          top_k: int = 5,
                          filters: Optional[Dict] = None) -> tuple:
        """
        Retrieve relevant documents from ChromaDB using semantic search.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            filters: Optional metadata filters (e.g., {"disease_category": "Pneumonia"})
            
        Returns:
            Tuple of (retrieved_documents, retrieval_time)
            
        Example:
            docs, time = retrieve_documents("symptoms of pneumonia", top_k=5)
        """
        retrieval_start = time.time()
        
        # Encode query with E5
        query_embedding = self.encode_query(query)
        
        # Build ChromaDB query parameters
        query_params = {
            'query_embeddings': [query_embedding.tolist()],
            'n_results': top_k
        }
        
        # Add metadata filters if provided
        if filters:
            query_params['where'] = filters
        
        # Query ChromaDB
        results = self.collection.query(**query_params)
        
        # Format results with metadata
        retrieved_docs = []
        
        for i, (doc_id, document, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Convert distance to similarity (for cosine: similarity = 1 - distance)
            similarity = 1 - distance
            
            retrieved_docs.append({
                'rank': i + 1,
                'doc_id': doc_id,
                'text': document,
                'similarity': similarity,
                'distance': distance,
                'metadata': metadata
            })
        
        retrieval_time = time.time() - retrieval_start
        
        return retrieved_docs, retrieval_time
    
    def format_context(self, documents: List[Dict], max_tokens: int = None) -> str:
        """
        Format retrieved documents into a context string for the LLM.
        
        Args:
            documents: List of retrieved documents with metadata
            max_tokens: Maximum tokens for context (default: self.max_context_tokens)
            
        Returns:
            Formatted context string with document boundaries
        """
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        context_parts = []
        current_tokens = 0
        
        for doc in documents:
            # Format each document with clear boundaries
            doc_text = f"""
Document {doc['rank']} [Disease: {doc['metadata'].get('disease_category', 'Unknown')}]:
{doc['text']}
---
"""
            
            # Estimate tokens (rough: 1 token ≈ 4 characters)
            doc_tokens = len(doc_text) // 4
            
            # Check if adding this document exceeds limit
            if current_tokens + doc_tokens > max_tokens:
                # Truncate if we have some space left
                remaining_chars = (max_tokens - current_tokens) * 4
                if remaining_chars > 100:  # Only add if meaningful
                    doc_text = doc_text[:remaining_chars] + "...\n---\n"
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            current_tokens += doc_tokens
        
        return "\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create Mistral-formatted prompt with clinical system instructions.
        
        Mistral Instruction Format:
        <s>[INST] {system_instruction}

        {context}

        {query} [/INST]
        
        Args:
            query: User query
            context: Formatted context from retrieved documents
            
        Returns:
            Complete formatted prompt for Mistral
        """
        # Clinical domain system instructions
        system_instruction = """You are a clinical AI assistant with expertise in medical diagnostics and patient care. Your role is to provide accurate, evidence-based answers using the provided clinical notes.

Guidelines:
- Base your answers strictly on the provided clinical context
- Cite specific information from the documents when possible
- Use clear, professional medical terminology
- If the context doesn't contain sufficient information, clearly state what's missing
- Never fabricate medical information or make unsupported claims
- Consider differential diagnoses when appropriate
- Acknowledge uncertainty when present in the data"""

        # Combine into Mistral instruction format
        prompt = f"""<s>[INST] {system_instruction}

Clinical Context from Patient Records:
{context}

Based on the clinical context above, answer the following question:

Question: {query}

Provide a clear, structured, evidence-based answer. [/INST]"""
        
        return prompt
    
    def generate_answer(self,
                       query: str,
                       top_k: int = 5,
                       filters: Optional[Dict] = None,
                       temperature: Optional[float] = None,
                       max_new_tokens: Optional[int] = None,
                       show_progress: bool = False) -> Dict:
        """
        Complete RAG pipeline: retrieve documents, format context, and generate answer.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            filters: Optional metadata filters (e.g., disease category)
            temperature: Generation temperature (default: 0.7)
            max_new_tokens: Maximum tokens to generate (default: 256)
            show_progress: Print progress messages
            
        Returns:
            Dictionary with:
            - query: Original query
            - answer: Generated answer text
            - sources: Retrieved source documents
            - metadata: Timing and token statistics
            
        Example:
            result = pipeline.generate_answer(
                "What are symptoms of pneumonia?",
                top_k=5,
                temperature=0.7
            )
            print(result['answer'])
        """
        pipeline_start = time.time()
        
        try:
            # ============================================================
            # STEP 1: RETRIEVAL
            # ============================================================
            if show_progress:
                print("  [1/4] Retrieving documents...")
            
            documents, retrieval_time = self.retrieve_documents(
                query, 
                top_k=top_k, 
                filters=filters
            )
            
            # Handle no results
            if not documents:
                return {
                    'query': query,
                    'answer': "No relevant documents found in the database. Please try a different query or check your filters.",
                    'sources': [],
                    'metadata': {
                        'retrieval_time': retrieval_time,
                        'generation_time': 0,
                        'total_time': time.time() - pipeline_start,
                        'error': 'No documents retrieved',
                        'success': False
                    }
                }
            
            # ============================================================
            # STEP 2: FORMAT CONTEXT
            # ============================================================
            if show_progress:
                print("  [2/4] Formatting context...")
            
            context = self.format_context(documents)
            
            # ============================================================
            # STEP 3: CREATE PROMPT
            # ============================================================
            if show_progress:
                print("  [3/4] Creating prompt...")
            
            prompt = self.create_prompt(query, context)
            
            # ============================================================
            # STEP 4: GENERATE ANSWER
            # ============================================================
            if show_progress:
                print("  [4/4] Generating answer...")
            
            generation_start = time.time()
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length
            ).to(self.model.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            # Update generation config with overrides
            gen_config = self.default_generation_config.copy()
            if temperature is not None:
                gen_config['temperature'] = temperature
            if max_new_tokens is not None:
                gen_config['max_new_tokens'] = max_new_tokens
            
            # Generate with no gradient tracking (inference only)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_config
                )
            
            # Decode output (skip the input prompt)
            generated_text = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            
            generation_time = time.time() - generation_start
            output_length = len(outputs[0]) - input_length
            
            # ============================================================
            # STEP 5: CLEANUP & RETURN
            # ============================================================
            
            # Clean up GPU memory
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Prepare result
            total_time = time.time() - pipeline_start
            
            result = {
                'query': query,
                'answer': generated_text.strip(),
                'sources': documents,
                'metadata': {
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time,
                    'total_time': total_time,
                    'input_tokens': input_length,
                    'output_tokens': output_length,
                    'total_tokens': input_length + output_length,
                    'context_length': len(context),
                    'documents_retrieved': len(documents),
                    'model_name': "mistralai/Mistral-7B-Instruct-v0.2",
                    'temperature': gen_config['temperature'],
                    'max_new_tokens': gen_config['max_new_tokens'],
                    'success': True
                }
            }
            
            return result
            
        except RuntimeError as e:
            # Handle GPU out of memory
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                
                return {
                    'query': query,
                    'answer': "ERROR: GPU out of memory. Try reducing max_new_tokens or top_k.",
                    'sources': documents if 'documents' in locals() else [],
                    'metadata': {
                        'retrieval_time': retrieval_time if 'retrieval_time' in locals() else 0,
                        'generation_time': 0,
                        'total_time': time.time() - pipeline_start,
                        'error': 'GPU OOM',
                        'success': False
                    }
                }
            else:
                raise e
                
        except Exception as e:
            # Handle any other errors
            return {
                'query': query,
                'answer': f"ERROR: {str(e)}",
                'sources': [],
                'metadata': {
                    'retrieval_time': 0,
                    'generation_time': 0,
                    'total_time': time.time() - pipeline_start,
                    'error': str(e),
                    'success': False
                }
            }
    
    def get_statistics(self) -> Dict:
        """
        Get pipeline statistics (if tracking is implemented).
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            'model_name': "mistralai/Mistral-7B-Instruct-v0.2",
            'embedding_model': "intfloat/e5-small-v2",
            'collection_size': self.collection.count(),
            'max_context_tokens': self.max_context_tokens,
            'max_input_length': self.max_input_length
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cleanup_memory():
    """Force garbage collection and clear GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (1 token ≈ 4 characters).
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
    RAG Pipeline Module
    ===================
    
    This module provides the core RAG pipeline for clinical question answering.
    
    Example usage:
    
        from rag_pipeline import RAGPipelineMistral
        
        # Initialize pipeline (after loading models)
        pipeline = RAGPipelineMistral(
            chroma_collection=collection,
            embedding_model=e5_model,
            generation_model=mistral_model,
            tokenizer=mistral_tokenizer
        )
        
        # Generate answer
        result = pipeline.generate_answer(
            query="What are the symptoms of pneumonia?",
            top_k=5,
            temperature=0.7
        )
        
        print(result['answer'])
        print(f"Retrieved {len(result['sources'])} sources")
        print(f"Total time: {result['metadata']['total_time']:.2f}s")
    
    For deployment examples, see app.py
    """)
