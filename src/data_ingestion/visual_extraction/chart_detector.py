"""
Chart and image detection module for BCG Multi-Agent & Multimodal AI Platform.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import pdf2image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class VisualElement:
    """Data class for representing visual elements extracted from documents."""
    element_id: str
    page_num: int
    element_type: str  # 'chart', 'graph', 'table', 'image', etc.
    image: Image.Image
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    confidence_score: float = 0.0
    extracted_data: Optional[Dict] = None
    description: Optional[str] = None
    source_path: Optional[str] = None
    
    def save(self, output_dir: Union[str, Path], filename: Optional[str] = None) -> str:
        """
        Save the visual element image to disk.
        
        Args:
            output_dir: Directory to save the image.
            filename: Custom filename (default: auto-generated based on element properties).
            
        Returns:
            Path to the saved image.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"{self.element_type}_{self.element_id}_page{self.page_num}.png"
        
        output_path = output_dir / filename
        self.image.save(output_path)
        return str(output_path)


class ChartDetector:
    """
    Detects and extracts charts, graphs, and other visual elements from PDFs.
    
    This class provides methods to identify visual elements in PDF documents
    and extract them as images for further processing.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the chart detector.
        
        Args:
            confidence_threshold: Minimum confidence score to consider a visual element.
        """
        self.confidence_threshold = confidence_threshold
        # In a real implementation, we might load ML models here
        # For now, we'll use a simple approach based on image analysis
    
    def extract_images_from_pdf(self, pdf_path: Union[str, Path]) -> List[Image.Image]:
        """
        Extract images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of PIL Image objects.
        """
        logger.info(f"Extracting images from {pdf_path}")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            # Convert PDF pages to images
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=300,
                fmt="png",
            )
            
            logger.debug(f"Extracted {len(images)} page images from {pdf_path}")
            return images
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {str(e)}")
            raise
    
    def _is_likely_chart(self, image: Image.Image) -> Tuple[bool, float]:
        """
        Determine if an image likely contains a chart.
        
        This is a simplified implementation. In a production system, this would 
        use a trained ML model to classify image content.
        
        Args:
            image: The PIL Image to analyze.
            
        Returns:
            Tuple of (is_chart, confidence_score)
        """
        # Convert to grayscale for analysis
        gray_image = image.convert('L')
        img_array = np.array(gray_image)
        
        # Simple heuristics to detect charts (this is just an example)
        # 1. Check for pixel variance (charts usually have distinct patterns)
        variance = np.var(img_array)
        
        # 2. Check for horizontal/vertical lines (common in charts)
        # This is a very simplified approach
        h_edges = np.diff(img_array, axis=1)
        v_edges = np.diff(img_array, axis=0)
        
        strong_h_edges = np.sum(np.abs(h_edges) > 50)
        strong_v_edges = np.sum(np.abs(v_edges) > 50)
        
        # Calculate a simple confidence score
        # This would be replaced by an actual model prediction in production
        edge_density = (strong_h_edges + strong_v_edges) / (img_array.shape[0] * img_array.shape[1])
        confidence = min(1.0, max(0.0, edge_density * 10 + variance / 1000))
        
        return confidence > self.confidence_threshold, confidence
    
    def detect_charts(self, pdf_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> List[VisualElement]:
        """
        Detect and extract charts from a PDF.
        
        Args:
            pdf_path: Path to the PDF file.
            output_dir: Optional directory to save extracted charts.
            
        Returns:
            List of VisualElement objects representing detected charts.
        """
        logger.info(f"Detecting charts in {pdf_path}")
        
        # Extract all page images from the PDF
        page_images = self.extract_images_from_pdf(pdf_path)
        
        visual_elements = []
        for page_num, page_image in enumerate(page_images, 1):
            # For simplicity, we're treating each page as a potential chart
            # In a real implementation, we would segment the page and analyze each region
            
            is_chart, confidence = self._is_likely_chart(page_image)
            if is_chart:
                element_id = f"chart_{page_num}_{len(visual_elements)}"
                element = VisualElement(
                    element_id=element_id,
                    page_num=page_num,
                    element_type="chart",  # A more sophisticated system would classify the chart type
                    image=page_image,
                    confidence_score=confidence,
                    source_path=str(pdf_path),
                    description=f"Potential chart detected on page {page_num}",
                )
                
                visual_elements.append(element)
                
                # Save the image if output_dir is provided
                if output_dir:
                    element.save(output_dir)
        
        logger.debug(f"Detected {len(visual_elements)} potential charts in {pdf_path}")
        return visual_elements