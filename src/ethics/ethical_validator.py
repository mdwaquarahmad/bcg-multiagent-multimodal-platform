"""
Ethical validator for BCG Multi-Agent & Multimodal AI Platform.

This module implements an ethical validator that ensures outputs from the
platform adhere to ethical guidelines and responsible AI principles based
on IEEE Ethically Aligned Design framework.
"""
import logging
import re
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class EthicalPrinciple(str, Enum):
    """Ethical principles from IEEE Ethically Aligned Design framework."""
    HUMAN_RIGHTS = "human_rights"
    WELL_BEING = "well_being"
    ACCOUNTABILITY = "accountability"
    TRANSPARENCY = "transparency"
    AWARENESS_OF_MISUSE = "awareness_of_misuse"
    COMPETENCE = "competence"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    SAFETY = "safety"
    SUSTAINABILITY = "sustainability"

class EthicalAssessment(BaseModel):
    """Assessment model for ethical validation results."""
    principle: EthicalPrinciple = Field(..., description="The ethical principle being assessed")
    score: float = Field(..., description="Assessment score between 0.0 and 1.0", ge=0.0, le=1.0)
    concerns: List[str] = Field(default_factory=list, description="Specific concerns identified")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")

class EthicalValidationResult(BaseModel):
    """Results of ethical validation."""
    content: str = Field(..., description="The content that was validated")
    valid: bool = Field(..., description="Whether the content passes ethical validation")
    overall_score: float = Field(..., description="Overall ethical score between 0.0 and 1.0", ge=0.0, le=1.0)
    assessments: List[EthicalAssessment] = Field(..., description="Assessment for each ethical principle")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Overall suggestions for improvement")

class EthicalValidator:
    """
    Ethical validator that ensures outputs adhere to ethical guidelines.
    
    This validator is based on the IEEE Ethically Aligned Design framework
    and ensures that outputs from the BCG Multi-Agent & Multimodal AI Platform
    adhere to responsible AI principles.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        principles: Optional[List[EthicalPrinciple]] = None,
        threshold: float = 0.7,
        verbose: bool = False,
    ):
        """
        Initialize the ethical validator.
        
        Args:
            llm: Language model for ethical assessment.
            principles: List of ethical principles to check (default: all).
            threshold: Threshold score for passing validation (0.0-1.0).
            verbose: Whether to log verbose information.
        """
        self.llm = llm
        self.principles = principles or list(EthicalPrinciple)
        self.threshold = threshold
        self.verbose = verbose
        
        # IEEE Ethically Aligned Design descriptions
        self.principle_descriptions = {
            EthicalPrinciple.HUMAN_RIGHTS: "Respect and protect human rights and ensure AI systems don't violate human dignity",
            EthicalPrinciple.WELL_BEING: "Prioritize human well-being in AI system design and use",
            EthicalPrinciple.ACCOUNTABILITY: "Ensure clear responsibility and accountability for AI systems",
            EthicalPrinciple.TRANSPARENCY: "Ensure AI systems operate transparently and their decisions are explainable",
            EthicalPrinciple.AWARENESS_OF_MISUSE: "Awareness and prevention of potential misuse of AI systems",
            EthicalPrinciple.COMPETENCE: "Ensure AI systems operate within their competence boundaries",
            EthicalPrinciple.PRIVACY: "Respect user privacy and data ownership",
            EthicalPrinciple.FAIRNESS: "Ensure AI systems are fair and avoid bias or discrimination",
            EthicalPrinciple.SAFETY: "Ensure AI systems operate safely and securely",
            EthicalPrinciple.SUSTAINABILITY: "Consider long-term effects of AI systems on society and the environment",
        }
        
        logger.info(f"Ethical validator initialized with {len(self.principles)} principles")
    
    def _assess_principle(self, content: str, principle: EthicalPrinciple) -> EthicalAssessment:
        """
        Assess content against a specific ethical principle.
        
        Args:
            content: Content to assess.
            principle: Ethical principle to assess against.
            
        Returns:
            Assessment for the principle.
        """
        principle_name = principle.value
        principle_description = self.principle_descriptions[principle]
        
        prompt = f"""
        You are an ethical AI validator, please evaluate the following content against the ethical principle of {principle_name.replace('_', ' ').title()}.

Principle description: {principle_description}

Content to evaluate:
---
{content}
---

Please provide:
1. A score from 0.0 to 1.0 representing how well the content adheres to this principle (1.0 means perfect adherence, 0.0 means severe violation)
2. Any specific concerns related to this principle
3. Suggestions for improvement to better adhere to this principle

Format your response as a valid JSON object with the following structure:
{{
  "score": <float between 0.0 and 1.0>,
  "concerns": [<list of specific concerns>],
  "suggestions": [<list of suggestions for improvement>]
}}

Ensure your response is a valid JSON object using double quotes for strings and containing only the JSON object, nothing else."""
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            
            # Extract JSON from the response
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            
            if json_match:
                import json
                assessment_dict = json.loads(json_match.group(1))
                
                # Create EthicalAssessment object
                assessment = EthicalAssessment(
                    principle=principle,
                    score=float(assessment_dict.get("score", 0.0)),
                    concerns=assessment_dict.get("concerns", []),
                    suggestions=assessment_dict.get("suggestions", []),
                )
                
                if self.verbose:
                    logger.info(f"Assessment for {principle_name}: Score={assessment.score}")
                
                return assessment
            else:
                logger.warning(f"Could not extract valid JSON from response for {principle_name}")
                # Return default assessment with low score
                return EthicalAssessment(
                    principle=principle,
                    score=0.5,
                    concerns=["Could not properly assess this principle"],
                    suggestions=["Review content manually for this principle"],
                )
        except Exception as e:
            logger.error(f"Error assessing principle {principle_name}: {str(e)}")
            # Return default assessment with low score
            return EthicalAssessment(
                principle=principle,
                score=0.5,
                concerns=[f"Error during assessment: {str(e)}"],
                suggestions=["Review content manually for this principle"],
            )
    
    def validate(self, content: str) -> EthicalValidationResult:
        """
        Validate content against ethical principles.
        
        Args:
            content: Content to validate.
            
        Returns:
            Validation result.
        """
        logger.info(f"Validating content against {len(self.principles)} ethical principles")
        
        # Assess each principle
        assessments = []
        for principle in self.principles:
            assessment = self._assess_principle(content, principle)
            assessments.append(assessment)
        
        # Calculate overall score (average of all principle scores)
        overall_score = sum(assessment.score for assessment in assessments) / len(assessments)
        
        # Determine if content is valid based on threshold
        valid = overall_score >= self.threshold
        
        # Collect improvement suggestions
        improvement_suggestions = []
        for assessment in assessments:
            if assessment.score < self.threshold:
                for suggestion in assessment.suggestions:
                    if suggestion not in improvement_suggestions:
                        improvement_suggestions.append(suggestion)
        
        # Create validation result
        result = EthicalValidationResult(
            content=content,
            valid=valid,
            overall_score=overall_score,
            assessments=assessments,
            improvement_suggestions=improvement_suggestions,
        )
        
        if self.verbose:
            logger.info(f"Validation result: valid={valid}, overall_score={overall_score:.2f}")
            if not valid:
                logger.info(f"Improvement suggestions: {improvement_suggestions}")
        
        return result
    
    def get_improvement_prompt(self, validation_result: EthicalValidationResult) -> str:
        """
        Generate a prompt for improving content based on validation results.
        
        Args:
            validation_result: Validation result.
            
        Returns:
            Improvement prompt for the LLM.
        """
        if validation_result.valid:
            return "The content meets ethical standards and does not require improvements."
        
        # Identify principles with low scores
        low_scoring_principles = [
            assessment for assessment in validation_result.assessments 
            if assessment.score < self.threshold
        ]
        
        # Create improvement prompt
        prompt = "Please improve the following content to address these ethical concerns:\n\n"
        
        for principle in low_scoring_principles:
            principle_name = principle.principle.value.replace('_', ' ').title()
            prompt += f"**{principle_name}** (Score: {principle.score:.2f}):\n"
            
            if principle.concerns:
                prompt += "Concerns:\n"
                for concern in principle.concerns:
                    prompt += f"- {concern}\n"
            
            if principle.suggestions:
                prompt += "Suggestions:\n"
                for suggestion in principle.suggestions:
                    prompt += f"- {suggestion}\n"
            
            prompt += "\n"
        
        prompt += "Original Content:\n\n"
        prompt += validation_result.content
        
        return prompt
    
    def improve_content(self, validation_result: EthicalValidationResult) -> str:
        """
        Improve content based on validation results.
        
        Args:
            validation_result: Validation result.
            
        Returns:
            Improved content.
        """
        if validation_result.valid:
            logger.info("Content already meets ethical standards")
            return validation_result.content
        
        logger.info("Generating improved content based on ethical validation")
        
        # Generate improvement prompt
        improvement_prompt = self.get_improvement_prompt(validation_result)
        
        # Use LLM to improve content
        try:
            response = self.llm.invoke(improvement_prompt)
            improved_content = response.content if hasattr(response, "content") else str(response)
            
            logger.info("Successfully generated improved content")
            return improved_content
        except Exception as e:
            logger.error(f"Error improving content: {str(e)}")
            return validation_result.content