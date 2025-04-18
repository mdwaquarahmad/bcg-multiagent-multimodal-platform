"""
Vector store for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages a vector store for efficient similarity search of text embeddings.
    
    This class provides methods to store, retrieve, and search documents 
    using their vector embeddings.
    """
    
    def __init__(
        self,
        embedding_model: Embeddings,
        persist_directory: Union[str, Path],
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: The embedding model to use.
            persist_directory: Directory to persist the vector store.
            collection_name: Optional name for the vector store collection.
        """
        self.embedding_model = embedding_model
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name or "bcg_sustainability_reports"
        
        # Ensure the persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing vector store at {self.persist_directory} with collection '{self.collection_name}'")
        
        # Initialize the vector store
        self.db = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
        )
        
        logger.info(f"Vector store initialized with {self.db._collection.count()} documents")
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of text strings to add.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of IDs for the texts.
            
        Returns:
            List of IDs of the added texts.
        """
        try:
            logger.info(f"Adding {len(texts)} texts to vector store")
            return self.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add Documents to the vector store.
        
        Args:
            documents: List of Document objects to add.
            
        Returns:
            List of IDs of the added documents.
        """
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            return self.db.add_documents(documents=documents)
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The query text.
            k: Number of results to return.
            filter: Optional metadata filter.
            fetch_k: Optional number of documents to fetch (to apply MMR).
            
        Returns:
            List of tuples of (document, similarity score).
        """
        try:
            logger.info(f"Searching for '{query}' with k={k}")
            results = self.db.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter,
                fetch_k=fetch_k or (k * 4),
            )
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise
    
    def search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents using a vector embedding.
        
        Args:
            embedding: The query embedding.
            k: Number of results to return.
            filter: Optional metadata filter.
            
        Returns:
            List of tuples of (document, similarity score).
        """
        try:
            logger.info(f"Searching by vector with k={k}")
            docs_and_scores = self.db.similarity_search_by_vector_with_relevance_scores(
                embedding=embedding,
                k=k,
                filter=filter,
            )
            return docs_and_scores
        except Exception as e:
            logger.error(f"Error searching vector store by vector: {str(e)}")
            raise
    
    def search_mmr(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Search for documents using Maximal Marginal Relevance (MMR).
        
        MMR optimizes for similarity to the query AND diversity among results.
        
        Args:
            query: The query text.
            k: Number of results to return.
            fetch_k: Number of documents to fetch before applying MMR.
            lambda_mult: Controls trade-off between relevance and diversity (0-1).
            filter: Optional metadata filter.
            
        Returns:
            List of Document objects.
        """
        try:
            logger.info(f"Performing MMR search for '{query}' with k={k}")
            docs = self.db.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
            )
            return docs
        except Exception as e:
            logger.error(f"Error performing MMR search: {str(e)}")
            raise
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: The ID of the document to retrieve.
            
        Returns:
            The Document object if found, None otherwise.
        """
        try:
            logger.info(f"Retrieving document with ID '{doc_id}'")
            results = self.db.get(ids=[doc_id])
            
            if results and results["documents"]:
                doc_text = results["documents"][0]
                metadata = results["metadatas"][0] if results["metadatas"] else {}
                
                return Document(page_content=doc_text, metadata=metadata)
            
            return None
        except Exception as e:
            logger.error(f"Error retrieving document by ID: {str(e)}")
            raise
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete documents from the vector store by ID.
        
        Args:
            ids: List of document IDs to delete.
        """
        try:
            logger.info(f"Deleting {len(ids)} documents from vector store")
            self.db.delete(ids=ids)
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    def persist(self) -> None:
        """Persist the vector store to disk."""
        try:
            logger.info(f"Persisting vector store to {self.persist_directory}")
            self.db.persist()
            logger.info("Vector store persisted successfully")
        except Exception as e:
            logger.error(f"Error persisting vector store: {str(e)}")
            raise
    
    def count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            Number of documents in the vector store.
        """
        return self.db._collection.count()