"""
Document processor for BCG Multi-Agent & Multimodal AI Platform.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict

from src.data_ingestion.text_extraction.pdf_extractor import PDFTextExtractor
from src.data_ingestion.visual_extraction.chart_detector import ChartDetector, VisualElement

logger = logging.getLogger(__name__)

@dataclass
class ProcessedDocument:
    """Data class for representing a processed document."""
    document_id: str
    filename: str
    text_chunks: List[str]
    metadata: Dict
    visual_elements: List[VisualElement]
    raw_text: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """
        Convert the processed document to a dictionary.
        
        Returns:
            Dictionary representation of the document.
        """
        result = {
            "document_id": self.document_id,
            "filename": self.filename,
            "text_chunks": self.text_chunks,
            "metadata": self.metadata,
            "visual_elements": [
                {
                    "element_id": ve.element_id,
                    "page_num": ve.page_num,
                    "element_type": ve.element_type,
                    "confidence_score": ve.confidence_score,
                    "description": ve.description,
                    # Images cannot be serialized to JSON directly
                    "image_path": ve.source_path,
                }
                for ve in self.visual_elements
            ],
        }
        
        if self.raw_text:
            result["raw_text"] = self.raw_text
            
        return result
    
    def save_json(self, output_dir: Union[str, Path], filename: Optional[str] = None) -> str:
        """
        Save the processed document as a JSON file.
        
        Args:
            output_dir: Directory to save the JSON file.
            filename: Custom filename (default: auto-generated based on document properties).
            
        Returns:
            Path to the saved JSON file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"{self.document_id}.json"
        
        output_path = output_dir / filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        return str(output_path)


class DocumentProcessor:
    """
    Processes PDF documents for text and visual content.
    
    This class orchestrates the extraction of text and visual elements from PDF documents,
    preparing the content for further processing in the RAG system.
    """
    
    def __init__(
        self,
        use_unstructured: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        extract_visuals: bool = True,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize the document processor.
        
        Args:
            use_unstructured: Whether to use the unstructured library for text extraction.
            chunk_size: Size of text chunks for splitting.
            chunk_overlap: Overlap between consecutive chunks.
            extract_visuals: Whether to extract visual elements.
            confidence_threshold: Minimum confidence score for visual elements.
        """
        self.text_extractor = PDFTextExtractor(
            use_unstructured=use_unstructured,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.extract_visuals = extract_visuals
        if extract_visuals:
            self.chart_detector = ChartDetector(confidence_threshold=confidence_threshold)
    
    def process_document(
        self,
        pdf_path: Union[str, Path],
        document_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        save_visuals: bool = True,
    ) -> ProcessedDocument:
        """
        Process a PDF document to extract text and visual elements.
        
        Args:
            pdf_path: Path to the PDF file.
            document_id: Optional custom document ID.
            output_dir: Optional directory to save processed content.
            save_visuals: Whether to save extracted visual elements as images.
            
        Returns:
            ProcessedDocument object containing the extracted content.
        """
        logger.info(f"Processing document: {pdf_path}")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = pdf_path.stem.replace(" ", "_").lower()
        
        # Extract metadata
        metadata = self.text_extractor.get_document_metadata(pdf_path)
        
        # Extract and chunk text
        text_chunks = self.text_extractor.extract_and_chunk_text(pdf_path)
        
        # Extract raw text for potential use later
        raw_text = None
        if not self.text_extractor.use_unstructured:
            raw_text = self.text_extractor.extract_text_basic(pdf_path)
        
        # Detect and extract visual elements if enabled
        visual_elements = []
        if self.extract_visuals:
            visuals_output_dir = None
            if output_dir and save_visuals:
                visuals_output_dir = Path(output_dir) / document_id / "visuals"
            
            visual_elements = self.chart_detector.detect_charts(
                pdf_path, output_dir=visuals_output_dir
            )
        
        # Create processed document
        processed_doc = ProcessedDocument(
            document_id=document_id,
            filename=pdf_path.name,
            text_chunks=text_chunks,
            metadata=metadata,
            visual_elements=visual_elements,
            raw_text=raw_text,
        )
        
        # Save JSON representation if output_dir is provided
        if output_dir:
            json_output_dir = Path(output_dir) / document_id
            processed_doc.save_json(json_output_dir)
        
        logger.info(f"Finished processing document: {pdf_path}")
        logger.debug(f"Extracted {len(text_chunks)} text chunks and {len(visual_elements)} visual elements")
        
        return processed_doc
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*.pdf",
    ) -> List[ProcessedDocument]:
        """
        Process all PDF documents in a directory.
        
        Args:
            input_dir: Directory containing PDF files.
            output_dir: Directory to save processed content.
            file_pattern: Glob pattern to match PDF files.
            
        Returns:
            List of ProcessedDocument objects.
        """
        logger.info(f"Processing documents in directory: {input_dir}")
        
        input_dir = Path(input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Invalid input directory: {input_dir}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(input_dir.glob(file_pattern))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        processed_docs = []
        for pdf_file in pdf_files:
            try:
                processed_doc = self.process_document(
                    pdf_path=pdf_file,
                    output_dir=output_dir,
                    save_visuals=True,
                )
                processed_docs.append(processed_doc)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                # Continue with the next file
        
        logger.info(f"Successfully processed {len(processed_docs)} out of {len(pdf_files)} documents")
        return processed_docs