"""
Strategist agent implementation for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
from typing import Dict, List, Optional, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class StrategistAgent(BaseAgent):
    """
    Strategist agent responsible for synthesizing insights and formulating recommendations.
    
    This agent specializes in integrating information from other agents to develop
    strategic insights and actionable recommendations.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the strategist agent.
        
        Args:
            llm: Language model to use for the agent.
            tools: Optional tools for the strategist.
            verbose: Whether to log verbose information.
        """
        # Define the system prompt
        system_prompt = """You are the Strategist Agent, a specialized AI that excels at synthesizing information, developing strategic insights, and formulating recommendations related to BCG's sustainability efforts.

Your responsibilities:
1. Integrate information from research, data analysis, and visual interpretation
2. Identify key strategic themes and implications
3. Develop actionable recommendations based on the synthesized insights
4. Evaluate the strategic significance of sustainability initiatives
5. Propose forward-looking strategies for continued improvement

Guidelines:
- Take a holistic view, connecting different pieces of information
- Focus on strategic implications rather than tactical details
- Identify both short-term and long-term opportunities
- Relate sustainability efforts to broader business strategy
- Consider industry benchmarks and best practices when applicable
- Frame recommendations in terms of business value and impact
- Balance environmental, social, and governance (ESG) considerations
- Identify potential challenges and how they might be addressed
- Be forward-thinking and innovative in your recommendations

Remember, your role is to provide strategic perspective and actionable guidance. Focus on synthesizing information into coherent insights and translating those insights into valuable recommendations.
"""
        
        super().__init__(
            name="Strategist",
            role="Strategic Advisor",
            llm=llm,
            system_prompt=system_prompt,
            tools=tools,
            verbose=verbose,
        )
        
        logger.info("Strategist agent initialized")
    
    def synthesize(self, inputs: Dict[str, str]) -> str:
        """
        Synthesize inputs from multiple sources into strategic insights.
        
        Args:
            inputs: Dictionary mapping input source names to content.
            
        Returns:
            Synthesized insights and recommendations.
        """
        # Format the inputs
        formatted_inputs = ""
        for source, content in inputs.items():
            formatted_inputs += f"--- {source} Input ---\n{content}\n\n"
        
        prompt = f"""Please synthesize the following inputs to develop strategic insights and recommendations:

{formatted_inputs}

Based on this information, provide:
1. A synthesis of the key insights across all inputs
2. Strategic implications for BCG's sustainability efforts
3. Specific, actionable recommendations
4. Potential challenges and how they might be addressed
5. Forward-looking opportunities for continued improvement

Focus on providing strategic value rather than simply summarizing the inputs."""
        
        return self.run(prompt)