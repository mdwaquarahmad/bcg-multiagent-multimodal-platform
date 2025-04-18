"""
Chart analysis tool for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
import io
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field

from PIL import Image
from langchain_core.tools import BaseTool, ToolException

logger = logging.getLogger(__name__)

class ChartAnalysisInput(BaseModel):
    """Input schema for the chart analysis tool."""
    image_path: Optional[str] = Field(None, description="Path to the image file to analyze")
    image_description: str = Field("", description="Description of what to look for in the image")

class ChartAnalysisTool(BaseTool):
    """
    Tool for analyzing charts and visual elements.
    
    This tool helps extract information and insights from charts, graphs,
    and other visual elements in the BCG Sustainability Reports.
    """
    
    name = "chart_analyzer"
    description = "Analyze charts, graphs, or visual elements to extract data and insights"
    args_schema = ChartAnalysisInput
    
    def __init__(
        self,
        visuals_directory: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the chart analysis tool.
        
        Args:
            visuals_directory: Directory containing visual elements to analyze.
        """
        super().__init__()
        
        self.visuals_directory = Path(visuals_directory) if visuals_directory else None
        self.available_images = self._scan_visuals_directory()
        
        logger.info(f"Chart analysis tool initialized with {len(self.available_images)} available images")
    
    def _scan_visuals_directory(self) -> Dict[str, Path]:
        """
        Scan the visuals directory for available images.
        
        Returns:
            Dictionary mapping image names to their file paths.
        """
        available_images = {}
        
        if self.visuals_directory and self.visuals_directory.exists():
            # Scan for image files recursively
            for image_path in self.visuals_directory.glob("**/*.png"):
                available_images[image_path.name] = image_path
            
            for image_path in self.visuals_directory.glob("**/*.jpg"):
                available_images[image_path.name] = image_path
        
        return available_images
    
    def _run(self, image_path: Optional[str] = None, image_description: str = "") -> str:
        """
        Analyze a chart or visual element.
        
        Args:
            image_path: Path to the image file to analyze.
            image_description: Description of what to look for in the image.
            
        Returns:
            Analysis results as a string.
        """
        logger.info(f"Analyzing chart/visual: {image_path}")
        
        # Check if an image path was provided
        if not image_path:
            # List available images if no specific image is requested
            return self._list_available_images(image_description)
        
        # Try to find the image
        image_file = None
        
        # Check if it's a direct path
        direct_path = Path(image_path)
        if direct_path.exists() and direct_path.is_file():
            image_file = direct_path
        
        # Check if it's a name in the available images
        elif image_path in self.available_images:
            image_file = self.available_images[image_path]
        
        # Check if it matches a partial name
        else:
            matches = [path for name, path in self.available_images.items() 
                      if image_path.lower() in name.lower()]
            if matches:
                image_file = matches[0]
        
        if not image_file:
            error_message = f"Image not found: {image_path}"
            logger.error(error_message)
            raise ToolException(error_message)
        
        try:
            # Load the image
            image = Image.open(image_file)
            
            # Basic image analysis
            width, height = image.size
            format_type = image.format
            mode = image.mode
            
            # Perform simple analysis based on the image
            analysis_result = f"Image Analysis for: {image_file.name}\n\n"
            analysis_result += f"Dimensions: {width} x {height} pixels\n"
            analysis_result += f"Format: {format_type}\n"
            analysis_result += f"Mode: {mode}\n\n"
            
            # Check if it's a color or grayscale image
            if mode == "RGB" or mode == "RGBA":
                analysis_result += "This is a color image.\n"
            elif mode == "L":
                analysis_result += "This is a grayscale image.\n"
            
            # Extract metadata if available
            if hasattr(image, "info") and image.info:
                analysis_result += "Metadata:\n"
                for key, value in image.info.items():
                    if isinstance(value, (str, int, float, bool)):
                        analysis_result += f"- {key}: {value}\n"
            
            # Basic chart type detection
            # This is a simplified approach - in a production system,
            # we would use more sophisticated image recognition
            
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Detect potential chart types based on simple heuristics
            if "chart" in image_file.name.lower() or "graph" in image_file.name.lower():
                # Check for bar chart indicators
                if "bar" in image_file.name.lower():
                    analysis_result += "\nThis appears to be a bar chart.\n"
                    analysis_result += "Bar charts are typically used to compare quantities across categories.\n"
                
                # Check for line chart indicators
                elif "line" in image_file.name.lower():
                    analysis_result += "\nThis appears to be a line chart.\n"
                    analysis_result += "Line charts are typically used to show trends over time or continuous data.\n"
                
                # Check for pie chart indicators
                elif "pie" in image_file.name.lower():
                    analysis_result += "\nThis appears to be a pie chart.\n"
                    analysis_result += "Pie charts are typically used to show proportions or percentages of a whole.\n"
                
                # Generic chart
                else:
                    analysis_result += "\nThis appears to be a chart or graph.\n"
                    analysis_result += "Without more advanced image recognition, I can only provide basic analysis.\n"
            
            # Respond to the specific image description if provided
            if image_description:
                analysis_result += f"\nRegarding your specific query '{image_description}':\n"
                analysis_result += "To provide a detailed answer about specific data points or trends shown in this visual, I would need more advanced chart recognition capabilities. However, I can note that this image is part of the BCG Sustainability Report, likely illustrating key metrics or initiatives related to their sustainability efforts.\n"
            
            logger.info(f"Chart analysis completed for {image_file.name}")
            return analysis_result
        except Exception as e:
            error_message = f"Error analyzing image: {str(e)}"
            logger.error(error_message)
            raise ToolException(error_message)
    
    def _list_available_images(self, description: str = "") -> str:
        """
        List all available images that the tool can analyze.
        
        Args:
            description: Optional description to filter images.
            
        Returns:
            Formatted string listing available images.
        """
        if not self.available_images:
            return "No images available for analysis. Please check the visuals directory."
        
        # Filter images by description if provided
        filtered_images = self.available_images
        if description:
            filtered_images = {name: path for name, path in self.available_images.items()
                             if description.lower() in name.lower()}
        
        if not filtered_images:
            return f"No images matching '{description}' found. Available images:\n" + \
                   "\n".join([f"- {name}" for name in self.available_images.keys()])
        
        # Format the list of available images
        result = f"Available images{' matching ' + description if description else ''}:\n\n"
        
        for i, (name, path) in enumerate(filtered_images.items(), 1):
            result += f"{i}. {name}\n"
        
        result += "\nTo analyze a specific image, provide its name or full path."
        return result