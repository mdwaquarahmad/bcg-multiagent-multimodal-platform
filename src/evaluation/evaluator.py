"""
Evaluator for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage

from src.evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)

class ResponseEvaluator:
    """
    Evaluator for assessing the quality of responses from the BCG Multi-Agent & Multimodal AI Platform.
    
    This class provides methods for evaluating responses against various metrics
    and using LLM-based assessment for more sophisticated evaluation.
    """
    
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        reference_data: Optional[Dict[str, str]] = None,
        log_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the response evaluator.
        
        Args:
            llm: Optional language model for LLM-based evaluation.
            reference_data: Optional dictionary mapping topics to reference text.
            log_dir: Optional directory for logging evaluation results.
        """
        self.llm = llm
        self.reference_data = reference_data or {}
        
        # Set up logging directory
        if log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = None
        
        logger.info("Response evaluator initialized")
    
    def evaluate(
        self,
        query: str,
        response: str,
        reference: Optional[str] = None,
        expected_topics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a response using automated metrics.
        
        Args:
            query: Original query.
            response: Generated response to evaluate.
            reference: Optional reference text (ground truth).
            expected_topics: Optional list of topics that should be covered.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info(f"Evaluating response for query: '{query}'")
        
        # If reference is not provided, try to find it in reference_data
        if reference is None:
            # Try to match query keywords with reference topics
            query_lower = query.lower()
            best_match = None
            best_match_score = 0
            
            for topic, ref_text in self.reference_data.items():
                topic_words = topic.lower().split()
                match_score = sum(1 for word in topic_words if word in query_lower)
                if match_score > best_match_score:
                    best_match = topic
                    best_match_score = match_score
            
            if best_match and best_match_score > 0:
                reference = self.reference_data[best_match]
                logger.info(f"Using reference for topic: {best_match}")
            else:
                # No good match found, use all references concatenated
                reference = "\n\n".join(self.reference_data.values())
                logger.info("Using combined references")
        
        # Run all evaluation metrics
        metrics = EvaluationMetrics.evaluate_all(
            query=query,
            response=response,
            reference=reference or "",
            expected_topics=expected_topics,
        )
        
        # Add metadata
        metrics["timestamp"] = datetime.now().isoformat()
        metrics["query"] = query
        
        # Log evaluation results if log_dir is specified
        if self.log_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"evaluation_{timestamp}.json"
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Evaluation results saved to {log_file}")
        
        return metrics
    
    def llm_evaluate(
        self,
        query: str,
        response: str,
        reference: Optional[str] = None,
        criteria: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a response using an LLM.
        
        Args:
            query: Original query.
            response: Generated response to evaluate.
            reference: Optional reference text (ground truth).
            criteria: Optional list of evaluation criteria.
            
        Returns:
            Dictionary with LLM-based evaluation results.
        """
        if self.llm is None:
            raise ValueError("LLM is required for LLM-based evaluation")
        
        logger.info(f"Performing LLM-based evaluation for query: '{query}'")
        
        # Default evaluation criteria
        default_criteria = [
            "Relevance to the query",
            "Factual accuracy",
            "Comprehensiveness",
            "Clarity and structure",
            "Use of citations",
            "Overall quality",
        ]
        
        criteria = criteria or default_criteria
        
        # Create evaluation prompt
        prompt = f"""As an expert evaluator, please assess the following response to a query about BCG's sustainability efforts. 

Query: "{query}"

Response to evaluate:
"""
        
        if reference:
            prompt += f"""
Reference information (ground truth):
{reference}

"""
        
        prompt += f"""
Response:
{response}

Please evaluate the response based on the following criteria:
{', '.join(criteria)}

For each criterion, provide a score from 1-10 and a brief explanation. Then provide an overall assessment and suggestions for improvement.

Format your evaluation as JSON with the following structure:
{{
  "criteria": {{
    "criterion1": {{
      "score": X,
      "explanation": "Your explanation here"
    }},
    ...
  }},
  "overall_score": X,
  "overall_assessment": "Your overall assessment here",
  "suggestions_for_improvement": ["Suggestion 1", "Suggestion 2", ...]
}}"""
        
        # Get evaluation from LLM
        llm_response = self.llm.invoke(prompt)
        
        # Extract JSON from response
        try:
            # Try to find JSON block in the response
            response_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                evaluation = json.loads(json_str)
            else:
                # If no JSON found, use the whole response
                logger.warning("No valid JSON found in LLM response, using raw response")
                evaluation = {"raw_response": response_text}
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response")
            evaluation = {"raw_response": str(llm_response)}
        
        # Add metadata
        evaluation["timestamp"] = datetime.now().isoformat()
        evaluation["query"] = query
        
        # Log evaluation results if log_dir is specified
        if self.log_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"llm_evaluation_{timestamp}.json"
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(evaluation, f, indent=2)
            
            logger.info(f"LLM evaluation results saved to {log_file}")
        
        return evaluation
    
    def evaluate_multiple(
        self,
        query: str,
        responses: List[str],
        reference: Optional[str] = None,
        expected_topics: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple responses to the same query.
        
        Args:
            query: Original query.
            responses: List of responses to evaluate.
            reference: Optional reference text (ground truth).
            expected_topics: Optional list of topics that should be covered.
            
        Returns:
            List of dictionaries with evaluation metrics for each response.
        """
        logger.info(f"Evaluating {len(responses)} responses for query: '{query}'")
        
        evaluations = []
        for i, response in enumerate(responses):
            logger.info(f"Evaluating response {i+1}/{len(responses)}")
            evaluation = self.evaluate(
                query=query,
                response=response,
                reference=reference,
                expected_topics=expected_topics,
            )
            evaluations.append(evaluation)
        
        return evaluations
    
    def compare_responses(
        self,
        query: str,
        responses: List[str],
        reference: Optional[str] = None,
        expected_topics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple responses to the same query.
        
        Args:
            query: Original query.
            responses: List of responses to evaluate.
            reference: Optional reference text (ground truth).
            expected_topics: Optional list of topics that should be covered.
            
        Returns:
            Dictionary with comparison results.
        """
        logger.info(f"Comparing {len(responses)} responses for query: '{query}'")
        
        # Evaluate all responses
        evaluations = self.evaluate_multiple(
            query=query,
            responses=responses,
            reference=reference,
            expected_topics=expected_topics,
        )
        
        # Compare the results
        comparison = {
            "query": query,
            "total_responses": len(responses),
            "evaluations": evaluations,
            "rankings": {},
        }
        
        # Rank responses by different metrics
        for metric in ["overall_score", "relevance", "factual_consistency", "hallucination_score"]:
            scores = [eval.get(metric, 0) for eval in evaluations]
            ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            comparison["rankings"][metric] = ranking
        
        # Find the best overall response
        overall_scores = [eval.get("overall_score", 0) for eval in evaluations]
        best_index = overall_scores.index(max(overall_scores))
        comparison["best_response_index"] = best_index
        comparison["best_response_score"] = overall_scores[best_index]
        
        logger.info(f"Best response is #{best_index+1} with score {overall_scores[best_index]:.2f}")
        
        return comparison
    
    def add_reference_data(self, topic: str, reference: str) -> None:
        """
        Add reference data for a topic.
        
        Args:
            topic: Topic name.
            reference: Reference text for the topic.
        """
        self.reference_data[topic] = reference
        logger.info(f"Added reference data for topic: {topic}")
    
    def load_reference_data(self, filepath: Union[str, Path]) -> None:
        """
        Load reference data from a JSON file.
        
        Args:
            filepath: Path to the JSON file.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Reference data file not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            reference_data = json.load(f)
        
        self.reference_data.update(reference_data)
        logger.info(f"Loaded reference data for {len(reference_data)} topics from {filepath}")