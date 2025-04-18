"""
Critic agent implementation for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
from typing import Dict, List, Optional, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class CriticAgent(BaseAgent):
    """
    Critic agent responsible for evaluating and improving outputs.
    
    This agent specializes in fact-checking, identifying gaps or inconsistencies,
    and suggesting improvements to ensure high-quality outputs.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the critic agent.
        
        Args:
            llm: Language model to use for the agent.
            tools: Optional tools for the critic.
            verbose: Whether to log verbose information.
        """
        # Define the system prompt
        system_prompt = """You are the Critic Agent, a specialized AI that excels at evaluating information, identifying issues, and improving the quality of outputs related to BCG's sustainability efforts.

Your responsibilities:
1. Fact-check information for accuracy and consistency
2. Identify gaps, inconsistencies, or weaknesses in analyses
3. Evaluate the quality and completeness of recommendations
4. Ensure outputs align with BCG's sustainability goals and values
5. Suggest specific improvements to enhance the overall quality

Guidelines:
- Be constructively critical, focusing on improvement rather than just criticism
- Verify claims against available information
- Check for logical consistency and sound reasoning
- Identify unstated assumptions or potential biases
- Assess whether conclusions follow from the evidence presented
- Evaluate whether recommendations are specific, actionable, and well-supported
- Consider whether important perspectives or alternatives have been overlooked
- Assess whether the outputs are clear, concise, and well-structured
- Check if the outputs address the original questions or objectives

Remember, your role is to ensure the highest quality outputs. Focus on identifying opportunities for improvement and suggesting specific enhancements.
"""
        
        super().__init__(
            name="Critic",
            role="Quality Evaluator",
            llm=llm,
            system_prompt=system_prompt,
            tools=tools,
            verbose=verbose,
        )
        
        logger.info("Critic agent initialized")
    
    def evaluate(self, content: str, context: Optional[str] = None) -> str:
        """
        Evaluate content and suggest improvements.
        
        Args:
            content: Content to evaluate.
            context: Optional context for the evaluation.
            
        Returns:
            Evaluation and suggested improvements.
        """
        prompt = f"""Please evaluate the following content and suggest specific improvements:

Content to Evaluate:
{content}

{f'Context:\n{context}' if context else ''}

Provide a thorough evaluation that includes:
1. Assessment of factual accuracy and consistency
2. Identification of gaps, inconsistencies, or weaknesses
3. Evaluation of the quality and completeness
4. Specific suggestions for improvement
5. Overall assessment of strengths and areas for enhancement

Focus on constructive criticism that can help improve the quality of the content."""
        
        return self.run(prompt)