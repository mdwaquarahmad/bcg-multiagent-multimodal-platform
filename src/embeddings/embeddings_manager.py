"""
Embeddings manager for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import uuid

from langchain_core.documents import Document

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStore
from src.data_ingestion.document_processor import ProcessedDocument

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """
    Manages the process of generating embeddings and storing them in a vector store.
    
    This class orchestrates the embedding generation and vector storage operations,
    providing a high-level interface for the RAG system.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_type: str = "local",
        vector_store_dir: Optional[Union[str, Path]] = None,
        collection_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        cache_folder: Optional[str] = None,
    ):
        """
        Initialize the embeddings manager.
        
        Args:
            model_name: Name of the embedding model to use.
            model_type: Type of embedding model ('local' or 'openai').
            vector_store_dir: Directory to persist the vector store.
            collection_name: Optional name for the vector store collection.
            openai_api_key: OpenAI API key (required for 'openai' model_type).
            cache_folder: Optional folder to cache embeddings models.
        """
        self.model_name = model_name
        self.model_type = model_type
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(
            model_name=model_name,
            model_type=model_type,
            openai_api_key=openai_api_key,
            cache_folder=cache_folder,
        )
        
        # Initialize vector store
        if vector_store_dir:
            self.vector_store_dir = Path(vector_store_dir)
        else:
            # Default to a subdirectory in the project's data directory
            project_root = Path(__file__).resolve().parent.parent.parent
            self.vector_store_dir = project_root / "data" / "embeddings" / "vector_store"
        
        self.vector_store = VectorStore(
            embedding_model=self.embedding_generator.get_embedding_model(),
            persist_directory=self.vector_store_dir,
            collection_name=collection_name,
        )
        
        logger.info(f"Embeddings manager initialized with model {model_name} ({model_type})")
    
    def process_document(self, processed_doc: ProcessedDocument) -> List[str]:
        """
        Process a document and add its text chunks to the vector store.
        
        Args:
            processed_doc: The processed document to add to the vector store.
            
        Returns:
            List of IDs of the added chunks.
        """
        logger.info(f"Processing document: {processed_doc.filename}")
        
        # Convert text chunks to documents with metadata
        documents = []
        
        for i, chunk in enumerate(processed_doc.text_chunks):
            # Create metadata for the chunk
            metadata = {
                "document_id": processed_doc.document_id,
                "filename": processed_doc.filename,
                "chunk_id": i,
                **{k: v for k, v in processed_doc.metadata.items() if isinstance(v, (str, int, float, bool))},
            }
            
            # Add information about visual elements if available
            if i < len(processed_doc.visual_elements):
                visual_element = processed_doc.visual_elements[i]
                metadata["has_visual"] = True
                metadata["visual_type"] = visual_element.element_type
                metadata["visual_page"] = visual_element.page_num
            else:
                metadata["has_visual"] = False
            
            # Create a document with the chunk and metadata
            doc = Document(page_content=chunk, metadata=metadata)
            documents.append(doc)
        
        # Add documents to vector store
        try:
            ids = self.vector_store.add_documents(documents)
            self.vector_store.persist()
            logger.info(f"Added {len(ids)} chunks from {processed_doc.filename} to vector store")
            return ids
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            raise
    
    def process_json_document(self, json_path: Union[str, Path]) -> List[str]:
        """
        Process a JSON document file and add its text chunks to the vector store.
        
        Args:
            json_path: Path to the JSON document file.
            
        Returns:
            List of IDs of the added chunks.
        """
        logger.info(f"Processing JSON document: {json_path}")
        
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)
            
            # Extract document information
            document_id = doc_data.get("document_id", json_path.stem)
            filename = doc_data.get("filename", json_path.name)
            text_chunks = doc_data.get("text_chunks", [])
            metadata = doc_data.get("metadata", {})
            
            # Convert visual elements if available
            visual_elements = []
            for visual_data in doc_data.get("visual_elements", []):
                visual_elements.append(visual_data)
            
            # Create documents with metadata
            documents = []
            
            for i, chunk in enumerate(text_chunks):
                # Create metadata for the chunk
                chunk_metadata = {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_id": i,
                    **{k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))},
                }
                
                # Add information about visual elements if available
                if i < len(visual_elements):
                    visual_element = visual_elements[i]
                    chunk_metadata["has_visual"] = True
                    chunk_metadata["visual_type"] = visual_element.get("element_type", "unknown")
                    chunk_metadata["visual_page"] = visual_element.get("page_num", 0)
                else:
                    chunk_metadata["has_visual"] = False
                
                # Create a document with the chunk and metadata
                doc = Document(page_content=chunk, metadata=chunk_metadata)
                documents.append(doc)
            
            # Add documents to vector store
            ids = self.vector_store.add_documents(documents)
            self.vector_store.persist()
            logger.info(f"Added {len(ids)} chunks from {filename} to vector store")
            return ids
        except Exception as e:
            logger.error(f"Error processing JSON document: {str(e)}")
            raise
    
    def process_directory(self, directory: Union[str, Path]) -> Dict[str, List[str]]:
        """
        Process all JSON document files in a directory and add them to the vector store.
        
        Args:
            directory: Directory containing JSON document files.
            
        Returns:
            Dictionary mapping filenames to lists of added chunk IDs.
        """
        logger.info(f"Processing directory: {directory}")
        
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory}")
        
        # Find all JSON files in the directory
        json_files = []
        
        # Check first-level directories (document_id directories)
        for doc_dir in directory.iterdir():
            if doc_dir.is_dir():
                # Look for JSON files in each document directory
                for json_file in doc_dir.glob("*.json"):
                    json_files.append(json_file)
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        # Process each JSON file
        results = {}
        for json_file in json_files:
            try:
                ids = self.process_json_document(json_file)
                results[json_file.name] = ids
            except Exception as e:
                logger.error(f"Error processing {json_file}: {str(e)}")
                # Continue with the next file
        
        logger.info(f"Successfully processed {len(results)} out of {len(json_files)} documents")
        return results
    
    def search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        use_mmr: bool = True,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> List[Document]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The query text.
            k: Number of results to return.
            filter: Optional metadata filter.
            use_mmr: Whether to use Maximal Marginal Relevance.
            fetch_k: Number of documents to fetch before applying MMR.
            lambda_mult: Controls trade-off between relevance and diversity (0-1).
            
        Returns:
            List of Document objects matching the query.
        """
        logger.info(f"Searching for '{query}' with k={k}, use_mmr={use_mmr}")
        
        try:
            if use_mmr:
                docs = self.vector_store.search_mmr(
                    query=query,
                    k=k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    filter=filter,
                )
                return docs
            else:
                docs_and_scores = self.vector_store.search(
                    query=query,
                    k=k,
                    filter=filter,
                    fetch_k=fetch_k,
                )
                return [doc for doc, _ in docs_and_scores]
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary of statistics.
        """
        try:
            document_count = self.vector_store.count()
            
            return {
                "document_count": document_count,
                "model_name": self.model_name,
                "model_type": self.model_type,
                "vector_store_dir": str(self.vector_store_dir),
                "embedding_dimension": self.embedding_generator.get_embedding_dimension(),
            }
        except Exception as e:
            logger.error(f"Error getting vector store statistics: {str(e)}")
            raise