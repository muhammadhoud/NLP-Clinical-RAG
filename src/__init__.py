"""
Clinical RAG Assistant - Source Package

This package contains the core RAG pipeline logic for the Clinical RAG Assistant.

Modules:
    rag_pipeline: Core RAG pipeline with retrieval and generation components

Author: Muhammad Houd
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Muhammad Houd"

# Import main classes for easier access
try:
    from .rag_pipeline import RAGPipelineMistral, ClinicalRetrieverChroma
    
    __all__ = [
        "RAGPipelineMistral",
        "ClinicalRetrieverChroma",
    ]
except ImportError:
    # Allow package to be imported even if dependencies aren't installed yet
    __all__ = []
