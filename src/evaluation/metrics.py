"""
Evaluation metrics for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """
    Collection of metrics for evaluating the BCG Multi-Agent & Multimodal AI Platform.
    
    This class provides various evaluation metrics to assess the quality,
    relevance, and factual accuracy of the platform's outputs.
    """
    
    @staticmethod
    def factual_consistency(response: str, reference: str) -> float:
        """
        Measure factual consistency between response and reference.
        
        This metric checks whether factual claims in the response are consistent
        with the information provided in the reference.
        
        Args:
            response: Generated response to evaluate.
            reference: Reference text (ground truth).
            
        Returns:
            Factual consistency score between 0 and 1.
        """
        logger.info("Calculating factual consistency")
        
        # Extract numerical facts from both texts
        def extract_numerical_facts(text):
            # Look for patterns like "X% reduction", "reduced by X%", "X million", etc.
            patterns = [
                r'(\d+(?:\.\d+)?)%',  # Percentages
                r'(\d+(?:\.\d+)?) percent',  # Percentages
                r'(\d+(?:\.\d+)?) million',  # Millions
                r'(\d+(?:\.\d+)?) billion',  # Billions
                r'\$(\d+(?:\.\d+)?)',  # Dollar amounts
                r'(\d{4})',  # Years
            ]
            
            facts = []
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    # Get some context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    facts.append((match.group(0), context))
            
            return facts
        
        response_facts = extract_numerical_facts(response)
        reference_facts = extract_numerical_facts(reference)
        
        if not response_facts:
            return 1.0  # No facts to verify
        
        # Count how many response facts appear in the reference
        consistent_facts = 0
        for fact, context in response_facts:
            for ref_fact, ref_context in reference_facts:
                if fact == ref_fact and any(word in ref_context for word in context.split() if len(word) > 4):
                    consistent_facts += 1
                    break
        
        consistency_score = consistent_facts / len(response_facts) if response_facts else 1.0
        
        logger.info(f"Factual consistency score: {consistency_score:.2f}")
        return consistency_score
    
    @staticmethod
    def citation_count(response: str) -> int:
        """
        Count the number of citations in the response.
        
        Args:
            response: Generated response to evaluate.
            
        Returns:
            Number of citations found.
        """
        # Look for citations in various formats
        citation_patterns = [
            r'\[Document \d+\]',  # [Document X]
            r'\[Source: [^\]]+\]',  # [Source: X]
            r'\(Source: [^\)]+\)',  # (Source: X)
            r'\(BCG \d{4}\)',  # (BCG YYYY)
        ]
        
        total_citations = 0
        for pattern in citation_patterns:
            citations = re.findall(pattern, response)
            total_citations += len(citations)
        
        logger.info(f"Citation count: {total_citations}")
        return total_citations
    
    @staticmethod
    def answer_relevance(query: str, response: str) -> float:
        """
        Measure the relevance of the response to the query.
        
        This metric uses a simple keyword matching approach to assess relevance.
        A more sophisticated approach would use semantic similarity.
        
        Args:
            query: Original query.
            response: Generated response to evaluate.
            
        Returns:
            Relevance score between 0 and 1.
        """
        logger.info("Calculating answer relevance")
        
        # Tokenize and clean the query
        def clean_text(text):
            text = text.lower()
            # Remove punctuation
            text = re.sub(r'[^\w\s]', ' ', text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            # Split into words
            return text.split()
        
        query_words = clean_text(query)
        response_words = clean_text(response)
        
        # Remove common English stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of'}
        query_keywords = [word for word in query_words if word not in stop_words and len(word) > 2]
        
        # Count how many query keywords appear in the response
        matches = sum(1 for word in query_keywords if word in response_words)
        relevance_score = matches / len(query_keywords) if query_keywords else 0.0
        
        logger.info(f"Answer relevance score: {relevance_score:.2f}")
        return relevance_score
    
    @staticmethod
    def comprehensiveness(response: str, expected_topics: List[str]) -> float:
        """
        Measure the comprehensiveness of the response.
        
        This metric checks whether the response covers all expected topics.
        
        Args:
            response: Generated response to evaluate.
            expected_topics: List of topics that should be covered.
            
        Returns:
            Comprehensiveness score between 0 and 1.
        """
        logger.info("Calculating comprehensiveness")
        
        covered_topics = 0
        for topic in expected_topics:
            if re.search(r'\b' + re.escape(topic) + r'\b', response, re.IGNORECASE):
                covered_topics += 1
        
        comprehensiveness_score = covered_topics / len(expected_topics) if expected_topics else 0.0
        
        logger.info(f"Comprehensiveness score: {comprehensiveness_score:.2f}")
        return comprehensiveness_score
    
    @staticmethod
    def response_length(response: str) -> Dict[str, int]:
        """
        Calculate various length metrics for the response.
        
        Args:
            response: Generated response to evaluate.
            
        Returns:
            Dictionary with various length metrics.
        """
        # Character count
        char_count = len(response)
        
        # Word count
        words = re.findall(r'\b\w+\b', response)
        word_count = len(words)
        
        # Sentence count
        sentences = re.split(r'[.!?]+', response)
        sentence_count = sum(1 for s in sentences if s.strip())
        
        # Paragraph count
        paragraphs = re.split(r'\n\s*\n', response)
        paragraph_count = sum(1 for p in paragraphs if p.strip())
        
        logger.info(f"Response length: {word_count} words, {sentence_count} sentences, {paragraph_count} paragraphs")
        
        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
        }
    
    @staticmethod
    def detect_hallucinations(response: str, reference: str) -> Tuple[List[str], float]:
        """
        Detect potential hallucinations in the response.
        
        This metric identifies statements that are presented as facts but are not
        supported by the reference information.
        
        Args:
            response: Generated response to evaluate.
            reference: Reference text (ground truth).
            
        Returns:
            Tuple of (list of potential hallucinations, hallucination score).
        """
        logger.info("Detecting potential hallucinations")
        
        # Split the response into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)
        
        # Identify sentences that make factual claims
        factual_claims = []
        for sentence in sentences:
            # Look for indicators of factual claims
            if (re.search(r'\b(is|are|was|were|has|have|had|will|would|according to|reported|stated)\b', sentence) and
                not re.search(r'\b(might|may|could|possibly|perhaps|probably|suggest)\b', sentence) and
                not re.search(r'If|Should|Would|Could|Can|May|Might|Must|Opinion', sentence, re.IGNORECASE)):
                factual_claims.append(sentence)
        
        # Check each factual claim against the reference
        potential_hallucinations = []
        for claim in factual_claims:
            # Extract key phrases (3-gram sliding window)
            words = re.findall(r'\b\w+\b', claim.lower())
            key_phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            
            # Check if any key phrase appears in the reference
            claim_supported = False
            for phrase in key_phrases:
                if phrase.lower() in reference.lower():
                    claim_supported = True
                    break
            
            if not claim_supported:
                potential_hallucinations.append(claim)
        
        hallucination_score = 1.0 - (len(potential_hallucinations) / len(factual_claims) if factual_claims else 0.0)
        
        logger.info(f"Detected {len(potential_hallucinations)} potential hallucinations out of {len(factual_claims)} factual claims")
        logger.info(f"Hallucination score: {hallucination_score:.2f}")
        
        return potential_hallucinations, hallucination_score
    
    @staticmethod
    def response_structure(response: str) -> Dict[str, Union[bool, float]]:
        """
        Evaluate the structure of the response.
        
        This metric assesses whether the response has a clear structure with
        sections, headings, bullet points, etc.
        
        Args:
            response: Generated response to evaluate.
            
        Returns:
            Dictionary with structure metrics.
        """
        logger.info("Evaluating response structure")
        
        # Check for markdown headings
        has_headings = bool(re.search(r'^#+\s+.+$', response, re.MULTILINE))
        
        # Check for bullet points
        has_bullets = bool(re.search(r'^\s*[\*\-\+•]\s+.+$', response, re.MULTILINE))
        
        # Check for numbered lists
        has_numbered_lists = bool(re.search(r'^\s*\d+\.\s+.+$', response, re.MULTILINE))
        
        # Check for paragraphs
        paragraphs = [p for p in re.split(r'\n\s*\n', response) if p.strip()]
        has_paragraphs = len(paragraphs) > 1
        
        # Count structure elements
        heading_count = len(re.findall(r'^#+\s+.+$', response, re.MULTILINE))
        bullet_count = len(re.findall(r'^\s*[\*\-\+•]\s+.+$', response, re.MULTILINE))
        numbered_item_count = len(re.findall(r'^\s*\d+\.\s+.+$', response, re.MULTILINE))
        
        # Calculate structure score based on presence of structural elements
        structure_score = sum([
            0.3 if has_headings else 0,
            0.3 if has_bullets or has_numbered_lists else 0,
            0.4 if has_paragraphs else 0
        ])
        
        logger.info(f"Structure score: {structure_score:.2f}")
        
        return {
            "has_headings": has_headings,
            "has_bullets": has_bullets,
            "has_numbered_lists": has_numbered_lists,
            "has_paragraphs": has_paragraphs,
            "heading_count": heading_count,
            "bullet_count": bullet_count,
            "numbered_item_count": numbered_item_count,
            "structure_score": structure_score
        }
    
    @staticmethod
    def evaluate_all(query: str, response: str, reference: str, expected_topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run all evaluation metrics on a response.
        
        Args:
            query: Original query.
            response: Generated response to evaluate.
            reference: Reference text (ground truth).
            expected_topics: Optional list of topics that should be covered.
            
        Returns:
            Dictionary with all evaluation metrics.
        """
        if expected_topics is None:
            expected_topics = []
        
        logger.info("Running all evaluation metrics")
        
        metrics = {}
        
        # Relevance
        metrics["relevance"] = EvaluationMetrics.answer_relevance(query, response)
        
        # Factual consistency
        metrics["factual_consistency"] = EvaluationMetrics.factual_consistency(response, reference)
        
        # Citations
        metrics["citation_count"] = EvaluationMetrics.citation_count(response)
        
        # Comprehensiveness
        metrics["comprehensiveness"] = EvaluationMetrics.comprehensiveness(response, expected_topics)
        
        # Length metrics
        metrics["length"] = EvaluationMetrics.response_length(response)
        
        # Hallucination detection
        hallucinations, score = EvaluationMetrics.detect_hallucinations(response, reference)
        metrics["hallucination_score"] = score
        metrics["potential_hallucinations"] = hallucinations
        
        # Structure
        metrics["structure"] = EvaluationMetrics.response_structure(response)
        
        # Calculate an overall quality score
        weights = {
            "relevance": 0.2,
            "factual_consistency": 0.3,
            "hallucination_score": 0.3,
            "comprehensiveness": 0.1,
            "structure_score": 0.1
        }
        
        overall_score = (
            weights["relevance"] * metrics["relevance"] +
            weights["factual_consistency"] * metrics["factual_consistency"] +
            weights["hallucination_score"] * metrics["hallucination_score"] +
            weights["comprehensiveness"] * metrics["comprehensiveness"] +
            weights["structure_score"] * metrics["structure"]["structure_score"]
        )
        
        metrics["overall_score"] = overall_score
        
        logger.info(f"Overall evaluation score: {overall_score:.2f}")
        
        return metrics