import numpy as np
import torch
from typing import List, Dict, Optional
import time
import gc

class RAGPipelineMistral:
    """Production RAG pipeline for clinical document retrieval and generation"""
    
    def __init__(self, chroma_collection, embedding_model, generation_model, 
                 tokenizer, documents_df=None, max_context_tokens=2000, max_input_length=4096):
        self.collection = chroma_collection
        self.embedding_model = embedding_model
        self.model = generation_model
        self.tokenizer = tokenizer
        self.documents_df = documents_df
        self.max_context_tokens = max_context_tokens
        self.max_input_length = max_input_length
        self.query_prefix = "query: "
        
        self.generation_config = {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
            'do_sample': True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query with E5 prefix"""
        prefixed_query = f"{self.query_prefix}{query}"
        with torch.no_grad():
            embedding = self.embedding_model.encode(
                prefixed_query,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
        return embedding
    
    def retrieve_documents(self, query: str, top_k: int = 5, filters: Optional[Dict] = None):
        """Retrieve documents from ChromaDB"""
        retrieval_start = time.time()
        query_embedding = self.encode_query(query)
        
        query_params = {
            'query_embeddings': [query_embedding.tolist()],
            'n_results': top_k
        }
        if filters:
            query_params['where'] = filters
        
        results = self.collection.query(**query_params)
        
        retrieved_docs = []
        for i, (doc_id, document, metadata, distance) in enumerate(zip(
            results['ids'][0], results['documents'][0], 
            results['metadatas'][0], results['distances'][0]
        )):
            retrieved_docs.append({
                'rank': i + 1,
                'doc_id': doc_id,
                'text': document,
                'similarity': 1 - distance,
                'distance': distance,
                'metadata': metadata
            })
        
        return retrieved_docs, time.time() - retrieval_start
    
    def format_context(self, documents: List[Dict], max_tokens: int = None) -> str:
        """Format retrieved documents into context string"""
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        context_parts = []
        current_tokens = 0
        
        for doc in documents:
            doc_text = f"""
Document {doc['rank']} [Disease: {doc['metadata'].get('disease_category', 'Unknown')}]:
{doc['text']}
---
"""
            doc_tokens = len(doc_text) // 4
            
            if current_tokens + doc_tokens > max_tokens:
                remaining_chars = (max_tokens - current_tokens) * 4
                if remaining_chars > 100:
                    doc_text = doc_text[:remaining_chars] + "...\n---\n"
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            current_tokens += doc_tokens
        
        return "\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """Create Mistral-formatted prompt"""
        system_instruction = """You are a clinical AI assistant with expertise in medical diagnostics and patient care. Your role is to provide accurate, evidence-based answers using the provided clinical notes.

Guidelines:
- Base your answers strictly on the provided clinical context
- Cite specific information from the documents when possible
- Use clear, professional medical terminology
- If the context doesn't contain sufficient information, clearly state what's missing
- Never fabricate medical information or make unsupported claims
- Consider differential diagnoses when appropriate
- Acknowledge uncertainty when present in the data"""
        
        prompt = f"""<s>[INST] {system_instruction}

Clinical Context from Patient Records:
{context}

Based on the clinical context above, answer the following question:

Question: {query}

Provide a clear, structured, evidence-based answer. [/INST]"""
        
        return prompt
    
    def generate_answer(self, query: str, top_k: int = 5, filters: Optional[Dict] = None, 
                       temperature: Optional[float] = None, max_new_tokens: Optional[int] = None, 
                       show_progress: bool = False) -> Dict:
        """Complete RAG pipeline: retrieve, format, and generate"""
        pipeline_start = time.time()
        
        try:
            # 1. Retrieval
            documents, retrieval_time = self.retrieve_documents(query, top_k=top_k, filters=filters)
            
            if not documents:
                return {
                    'query': query,
                    'answer': "No relevant documents found in the database.",
                    'sources': [],
                    'metadata': {
                        'retrieval_time': retrieval_time,
                        'generation_time': 0,
                        'total_time': time.time() - pipeline_start,
                        'error': 'No documents retrieved'
                    }
                }
            
            # 2. Format context
            context = self.format_context(documents)
            
            # 3. Create prompt
            prompt = self.create_prompt(query, context)
            
            # 4. Generate answer
            generation_start = time.time()
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length
            ).to(self.model.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            gen_config = self.generation_config.copy()
            if temperature is not None:
                gen_config['temperature'] = temperature
            if max_new_tokens is not None:
                gen_config['max_new_tokens'] = max_new_tokens
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_config)
            
            generated_text = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            
            generation_time = time.time() - generation_start
            output_length = len(outputs[0]) - input_length
            
            # Cleanup
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return {
                'query': query,
                'answer': generated_text.strip(),
                'sources': documents,
                'metadata': {
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time,
                    'total_time': time.time() - pipeline_start,
                    'input_tokens': input_length,
                    'output_tokens': output_length,
                    'total_tokens': input_length + output_length,
                    'success': True
                }
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                return {
                    'query': query,
                    'answer': "ERROR: GPU out of memory. Try reducing context size or max_new_tokens.",
                    'sources': documents if 'documents' in locals() else [],
                    'metadata': {
                        'retrieval_time': retrieval_time if 'retrieval_time' in locals() else 0,
                        'generation_time': 0,
                        'total_time': time.time() - pipeline_start,
                        'error': 'OOM during generation'
                    }
                }
            else:
                raise e
        except Exception as e:
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
