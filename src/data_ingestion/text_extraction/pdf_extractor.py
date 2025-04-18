"""
PDF text extraction module for BCG Multi-Agent & Multimodal AI Platform.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pypdf
from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class PDFTextExtractor:
    """
    Extracts and processes textual content from PDF documents.
    
    Provides methods for basic and advanced text extraction from PDF files,
    with options for structured extraction and text chunking.
    """
    
    def __init__(self, use_unstructured: bool = True, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF text extractor.
        
        Args:
            use_unstructured: Whether to use the unstructured library for advanced extraction.
            chunk_size: Size of text chunks for splitting.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.use_unstructured = use_unstructured
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
    def extract_text_basic(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text from a PDF file using PyPDF (basic extraction).
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Extracted text as a string.
        """
        logger.info(f"Extracting text from {pdf_path} using basic extraction")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                    
            logger.debug(f"Extracted {len(text)} characters from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            raise
    
    def extract_text_advanced(self, pdf_path: Union[str, Path]) -> List[Dict]:
        """
        Extract structured text from a PDF file using unstructured library.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of structured elements extracted from the PDF.
        """
        logger.info(f"Extracting text from {pdf_path} using advanced extraction")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            elements = partition_pdf(
                filename=str(pdf_path),
                extract_images_in_pdf=False,
                infer_table_structure=True,
            )
            
            # Convert elements to a more standardized format
            structured_elements = []
            for element in elements:
                element_type = type(element).__name__
                element_text = str(element)
                
                structured_elements.append({
                    "type": element_type,
                    "text": element_text,
                    "metadata": getattr(element, "metadata", {}),
                })
            
            logger.debug(f"Extracted {len(structured_elements)} elements from {pdf_path}")
            return structured_elements
        except Exception as e:
            logger.error(f"Error extracting structured text from {pdf_path}: {str(e)}")
            raise
    
    def extract_text(self, pdf_path: Union[str, Path]) -> Union[str, List[Dict]]:
        """
        Extract text from a PDF file using the configured method.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Extracted text as a string or list of structured elements.
        """
        if self.use_unstructured:
            return self.extract_text_advanced(pdf_path)
        else:
            return self.extract_text_basic(pdf_path)
    
    def extract_and_chunk_text(self, pdf_path: Union[str, Path]) -> List[str]:
        """
        Extract text from a PDF file and split into chunks.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of text chunks.
        """
        logger.info(f"Extracting and chunking text from {pdf_path}")
        
        if self.use_unstructured:
            elements = self.extract_text_advanced(pdf_path)
            text = "\n\n".join([element["text"] for element in elements])
        else:
            text = self.extract_text_basic(pdf_path)
        
        chunks = self.text_splitter.split_text(text)
        logger.debug(f"Split text into {len(chunks)} chunks")
        
        return chunks
    
    def get_document_metadata(self, pdf_path: Union[str, Path]) -> Dict:
        """
        Extract metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Dictionary of metadata.
        """
        logger.info(f"Extracting metadata from {pdf_path}")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                metadata = pdf_reader.metadata
                
                result = {
                    "filename": pdf_path.name,
                    "file_path": str(pdf_path),
                    "page_count": len(pdf_reader.pages),
                }
                
                # Add PDF metadata if available
                if metadata:
                    if metadata.title:
                        result["title"] = metadata.title
                    if metadata.author:
                        result["author"] = metadata.author
                    if metadata.subject:
                        result["subject"] = metadata.subject
                    if metadata.creator:
                        result["creator"] = metadata.creator
                    if metadata.producer:
                        result["producer"] = metadata.producer
                    if metadata.creation_date:
                        result["creation_date"] = str(metadata.creation_date)
                
                return result
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
            raise