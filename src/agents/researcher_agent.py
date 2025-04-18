"""
Researcher agent implementation for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
from typing import Dict, List, Optional, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from src.agents.base_agent import BaseAgent
from src.agents.tools.search_tool import SearchTool

logger = logging.getLogger(__name__)

class ResearcherAgent(BaseAgent):
    """
    Researcher agent responsible for gathering information.
    
    This agent specializes in collecting relevant information from both
    the BCG Sustainability Reports and external sources using search tools.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        additional_tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the researcher agent.
        
        Args:
            llm: Language model to use for the agent.
            additional_tools: Additional tools beyond the default search tool.
            verbose: Whether to log verbose information.
        """
        # Create default tools
        tools = [SearchTool()]
        
        # Add additional tools if provided
        if additional_tools:
            tools.extend(additional_tools)
        
        # Define the system prompt
        system_prompt = """You are the Researcher Agent, a specialized AI that excels at finding and gathering relevant information about BCG's sustainability initiatives and reports.

Your responsibilities:
1. Search for specific information within BCG sustainability reports
2. Gather data from external sources when needed
3. Research background context for sustainability topics
4. Verify facts and claims when possible
5. Provide comprehensive, accurate information to other agents

Guidelines:
- Be thorough and detail-oriented in your research
- Focus on factual information and reliable sources
- Prioritize information from official BCG sources whenever possible
- For external information, use the web_search tool to find up-to-date information
- Provide context for technical terms and sustainability concepts
- Indicate the source of each piece of information you provide
- Be clear about the time period relevant to each point

Remember, your role is to gather information, not to analyze or make recommendations - that will be done by other agents. Focus on providing comprehensive, accurate data that can inform further analysis.
"""
        
        super().__init__(
            name="Researcher",
            role="Information Gatherer",
            llm=llm,
            system_prompt=system_prompt,
            tools=tools,
            verbose=verbose,
        )
        
        logger.info("Researcher agent initialized")
    
    def research(self, topic: str) -> str:
        """
        Research a specific topic.
        
        Args:
            topic: Topic to research.
            
        Returns:
            Research findings as a string.
        """
        prompt = f"""Research the following topic related to BCG's sustainability efforts:

Topic: {topic}

Please provide a thorough compilation of information on this topic. Include factual data, key statistics, and important context. Use your tools if necessary to gather comprehensive information.

Format your findings as a well-organized report with clear sections and citations to sources."""
        
        return self.run(prompt)