"""
Analyst agent implementation for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
from typing import Dict, List, Optional, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from src.agents.base_agent import BaseAgent
from src.agents.tools.python_tool import PythonTool

logger = logging.getLogger(__name__)

class AnalystAgent(BaseAgent):
    """
    Analyst agent responsible for data analysis and insight generation.
    
    This agent specializes in analyzing data from BCG Sustainability Reports,
    generating insights, and identifying patterns and trends.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        additional_tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the analyst agent.
        
        Args:
            llm: Language model to use for the agent.
            additional_tools: Additional tools beyond the default Python tool.
            verbose: Whether to log verbose information.
        """
        # Create default tools
        tools = [PythonTool()]
        
        # Add additional tools if provided
        if additional_tools:
            tools.extend(additional_tools)
        
        # Define the system prompt
        system_prompt = """You are the Analyst Agent, a specialized AI that excels at data analysis, pattern recognition, and insight generation related to BCG's sustainability efforts.

Your responsibilities:
1. Analyze data from BCG sustainability reports to identify trends and patterns
2. Compare metrics across different years to identify progress and challenges
3. Generate data-driven insights about BCG's sustainability performance
4. Identify correlations between different sustainability initiatives
5. Use the python_executor tool for data analysis and visualization when needed

Guidelines:
- Be rigorous and methodical in your analysis
- Back up your insights with data and evidence
- Use quantitative methods where appropriate
- Look for trends over time and comparisons between different areas
- Focus on the most significant and meaningful patterns in the data
- Use visualizations (through Python) to illustrate your findings when helpful
- Highlight both positive trends and areas for potential improvement
- Be precise and specific about the metrics you're analyzing

Remember, your role is to analyze and provide insights based on data, not just to summarize information. Go beyond the surface level to identify meaningful patterns and implications.
"""
        
        super().__init__(
            name="Analyst",
            role="Data Analyst",
            llm=llm,
            system_prompt=system_prompt,
            tools=tools,
            verbose=verbose,
        )
        
        logger.info("Analyst agent initialized")
    
    def analyze(self, data: str, question: str) -> str:
        """
        Analyze data to answer a specific question.
        
        Args:
            data: Data to analyze.
            question: Question to answer through analysis.
            
        Returns:
            Analysis results as a string.
        """
        prompt = f"""Please analyze the following data to answer a specific question:

Data:
{data}

Question: {question}

Analyze this data thoroughly using your expertise in data analysis. Use the python_executor tool if you need to perform any calculations, statistical analysis, or create visualizations. 

Provide a comprehensive analysis that includes:
1. Key observations from the data
2. Trends or patterns identified
3. Specific insights that answer the question
4. Any additional relevant findings

Be precise and thorough in your analysis."""
        
        return self.run(prompt)