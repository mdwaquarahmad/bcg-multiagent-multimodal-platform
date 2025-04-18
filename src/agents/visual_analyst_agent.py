"""
Visual analyst agent implementation for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
from typing import Dict, List, Optional, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from src.agents.base_agent import BaseAgent
from src.agents.tools.chart_analysis_tool import ChartAnalysisTool

logger = logging.getLogger(__name__)

class VisualAnalystAgent(BaseAgent):
    """
    Visual analyst agent responsible for analyzing visual elements.
    
    This agent specializes in extracting information and insights from charts,
    graphs, and other visual elements in the BCG Sustainability Reports.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        visuals_directory: Optional[str] = None,
        additional_tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the visual analyst agent.
        
        Args:
            llm: Language model to use for the agent.
            visuals_directory: Directory containing visual elements to analyze.
            additional_tools: Additional tools beyond the default chart analysis tool.
            verbose: Whether to log verbose information.
        """
        # Create default tools
        tools = [ChartAnalysisTool(visuals_directory=visuals_directory)]
        
        # Add additional tools if provided
        if additional_tools:
            tools.extend(additional_tools)
        
        # Define the system prompt
        system_prompt = """You are the Visual Analyst Agent, a specialized AI that excels at analyzing and interpreting visual elements such as charts, graphs, and diagrams from BCG's sustainability reports.

Your responsibilities:
1. Extract data and information from charts and graphs
2. Interpret visual representations of BCG's sustainability metrics
3. Identify trends and patterns shown in visualizations
4. Explain the significance of visual data in the context of sustainability
5. Translate visual information into clear textual descriptions

Guidelines:
- Be methodical in your analysis of visual elements
- Describe what you see in detail, including the type of visualization
- Extract specific data points and values when possible
- Interpret the meaning of colors, sizes, and other visual encodings
- Explain what the visualization reveals about BCG's sustainability efforts
- Use the chart_analyzer tool to help analyze specific visual elements
- Provide context about how the visual data relates to BCG's broader sustainability goals
- Look for trends or comparisons shown in the visualization

Remember, your role is to make visual information accessible and meaningful. Focus on extracting and explaining the key insights presented in charts, graphs, and other visual elements from BCG's sustainability reports.
"""
        
        super().__init__(
            name="Visual Analyst",
            role="Visual Information Interpreter",
            llm=llm,
            system_prompt=system_prompt,
            tools=tools,
            verbose=verbose,
        )
        
        logger.info("Visual Analyst agent initialized")
    
    def analyze_visual(self, image_path: Optional[str] = None, description: str = "") -> str:
        """
        Analyze a visual element.
        
        Args:
            image_path: Path to the image to analyze.
            description: Description of what to look for in the image.
            
        Returns:
            Analysis results as a string.
        """
        if image_path:
            prompt = f"""Please analyze the following visual element:

Image: {image_path}
Looking for: {description}

Use the chart_analyzer tool to examine this visual element. Extract key data points, identify trends, and explain what this visualization reveals about BCG's sustainability efforts.

Provide a thorough analysis that includes:
1. Description of what type of visualization this is
2. Key data points or metrics shown
3. Apparent trends or patterns
4. Significance in the context of BCG's sustainability initiatives
5. Any limitations in the data presentation"""
        else:
            prompt = f"""Please list available visual elements that might contain information about:

Topic: {description}

Use the chart_analyzer tool to identify which visual elements might be relevant to this topic. Then provide a list of the most promising visuals that could contain information about this subject."""
        
        return self.run(prompt)