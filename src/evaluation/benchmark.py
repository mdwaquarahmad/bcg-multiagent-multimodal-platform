"""
Benchmark implementation for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

from src.evaluation.evaluator import ResponseEvaluator
from src.evaluation.metrics import AccuracyMetric, RelevanceMetric, CompletenessMetric, HallucinationMetric

logger = logging.getLogger(__name__)

class Benchmark:
    """
    Benchmark for evaluating the BCG Multi-Agent & Multimodal AI Platform.
    
    This class creates standardized test datasets and runs comprehensive
    evaluations to measure the performance of the platform.
    """
    
    def __init__(
        self,
        evaluator: ResponseEvaluator,
        benchmark_data_path: Optional[Union[str, Path]] = None,
        save_results: bool = True,
        results_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the benchmark.
        
        Args:
            evaluator: ResponseEvaluator instance.
            benchmark_data_path: Path to benchmark dataset (if None, will use default).
            save_results: Whether to save benchmark results to disk.
            results_dir: Directory to save results (if None, will use default).
        """
        self.evaluator = evaluator
        self.save_results = save_results
        
        # Set up paths
        if benchmark_data_path:
            self.benchmark_data_path = Path(benchmark_data_path)
        else:
            # Default to a benchmarks directory in the project
            project_root = Path(__file__).resolve().parent.parent.parent
            self.benchmark_data_path = project_root / "data" / "benchmarks" / "benchmark_dataset.json"
        
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            # Default to a results directory in the project
            project_root = Path(__file__).resolve().parent.parent.parent
            self.results_dir = project_root / "data" / "benchmarks" / "results"
        
        # Ensure results directory exists
        if self.save_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create benchmark dataset
        self.benchmark_data = self._load_or_create_benchmark_data()
        
        logger.info(f"Benchmark initialized with {len(self.benchmark_data)} test cases")
    
    def _load_or_create_benchmark_data(self) -> List[Dict[str, Any]]:
        """
        Load benchmark dataset or create default one if not found.
        
        Returns:
            List of benchmark test cases.
        """
        if self.benchmark_data_path.exists():
            try:
                with open(self.benchmark_data_path, "r", encoding="utf-8") as f:
                    benchmark_data = json.load(f)
                logger.info(f"Loaded benchmark dataset from {self.benchmark_data_path}")
                return benchmark_data
            except Exception as e:
                logger.error(f"Error loading benchmark dataset: {str(e)}")
        
        # Create default benchmark dataset
        logger.info(f"Creating default benchmark dataset at {self.benchmark_data_path}")
        benchmark_data = self._create_default_benchmark_data()
        
        # Save the default dataset
        self.benchmark_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.benchmark_data_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2)
        
        return benchmark_data
    
    def _create_default_benchmark_data(self) -> List[Dict[str, Any]]:
        """
        Create a default benchmark dataset with standard queries.
        
        Returns:
            List of benchmark test cases.
        """
        return [
            {
                "id": "factual-01",
                "query": "What are BCG's specific carbon emission reduction targets?",
                "category": "factual",
                "expected_sources": ["any"],
                "ground_truth": ["Carbon emission reduction targets"],
                "description": "Tests ability to extract specific factual information"
            },
            {
                "id": "factual-02",
                "query": "How many employees does BCG have according to their latest sustainability report?",
                "category": "factual",
                "expected_sources": ["BCG-2023-Annual-Sustainability-Report-April-2024.pdf"],
                "ground_truth": ["Employee count in 2023"],
                "description": "Tests ability to find specific numerical data from the latest report"
            },
            {
                "id": "factual-03",
                "query": "What percentage of BCG's Executive Committee are women?",
                "category": "factual",
                "expected_sources": ["BCG-2023-Annual-Sustainability-Report-April-2024.pdf"],
                "ground_truth": ["Percentage of women on Executive Committee"],
                "description": "Tests ability to find specific DEI-related metrics"
            },
            {
                "id": "analytical-01",
                "query": "How has BCG's approach to sustainability evolved over the past three years?",
                "category": "analytical",
                "expected_sources": ["multiple"],
                "ground_truth": ["Evolution of sustainability approach"],
                "description": "Tests ability to analyze trends across multiple reports"
            },
            {
                "id": "analytical-02",
                "query": "Compare BCG's diversity and inclusion efforts across the three sustainability reports.",
                "category": "analytical",
                "expected_sources": ["multiple"],
                "ground_truth": ["Comparison of D&I initiatives"],
                "description": "Tests comparative analysis capabilities"
            },
            {
                "id": "analytical-03",
                "query": "What progress has BCG made in reducing its environmental impact?",
                "category": "analytical",
                "expected_sources": ["multiple"],
                "ground_truth": ["Environmental impact reduction"],
                "description": "Tests ability to identify and quantify progress"
            },
            {
                "id": "strategic-01",
                "query": "What strategic recommendations would you make for BCG to improve its sustainability performance further?",
                "category": "strategic",
                "expected_sources": ["any"],
                "ground_truth": ["Strategic recommendations"],
                "description": "Tests strategic thinking and recommendation capabilities"
            },
            {
                "id": "strategic-02",
                "query": "How might BCG leverage its sustainability initiatives to create competitive advantage?",
                "category": "strategic",
                "expected_sources": ["any"],
                "ground_truth": ["Competitive advantage from sustainability"],
                "description": "Tests ability to connect sustainability to business strategy"
            },
            {
                "id": "visual-01",
                "query": "What do the charts in BCG's sustainability reports reveal about their carbon emissions?",
                "category": "visual",
                "expected_sources": ["any"],
                "ground_truth": ["Visual analysis of carbon emissions"],
                "description": "Tests visual analysis capabilities"
            },
            {
                "id": "complex-01",
                "query": "Analyze BCG's sustainability strategy, its implementation progress, and suggest future directions based on industry best practices.",
                "category": "complex",
                "expected_sources": ["multiple"],
                "ground_truth": ["Comprehensive sustainability analysis"],
                "description": "Tests handling of complex, multi-part queries"
            }
        ]
    
    def run_benchmark(self, system, verbose: bool = False) -> Dict[str, Any]:
        """
        Run benchmark tests against the provided system.
        
        Args:
            system: The multi-agent system to benchmark.
            verbose: Whether to print detailed results.
            
        Returns:
            Dictionary of benchmark results.
        """
        logger.info("Starting benchmark evaluation")
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_test_cases": len(self.benchmark_data),
            "metrics": {},
            "test_cases": []
        }
        
        # Track overall metrics
        accuracy_scores = []
        relevance_scores = []
        completeness_scores = []
        hallucination_scores = []
        
        # Run each test case
        for i, test_case in enumerate(self.benchmark_data):
            logger.info(f"Running test case {i+1}/{len(self.benchmark_data)}: {test_case['id']}")
            
            if verbose:
                print(f"\nTest case {i+1}/{len(self.benchmark_data)}: {test_case['id']}")
                print(f"Query: {test_case['query']}")
            
            # Process the query
            start_time = time.time()
            try:
                response = system.process_query(test_case["query"])
                processing_time = time.time() - start_time
                error = None
            except Exception as e:
                response = f"Error: {str(e)}"
                processing_time = time.time() - start_time
                error = str(e)
            
            # Evaluate the response
            if not error:
                evaluation_results = self.evaluator.evaluate_response(
                    query=test_case["query"],
                    response=response,
                    ground_truth=test_case.get("ground_truth", []),
                    category=test_case.get("category", "general"),
                    expected_sources=test_case.get("expected_sources", []),
                )
                
                # Extract scores
                accuracy_score = evaluation_results.get("accuracy", {}).get("score", 0)
                relevance_score = evaluation_results.get("relevance", {}).get("score", 0)
                completeness_score = evaluation_results.get("completeness", {}).get("score", 0)
                hallucination_score = evaluation_results.get("hallucination", {}).get("score", 0)
                
                # Track scores for overall metrics
                accuracy_scores.append(accuracy_score)
                relevance_scores.append(relevance_score)
                completeness_scores.append(completeness_score)
                hallucination_scores.append(hallucination_score)
            else:
                evaluation_results = {
                    "error": error,
                    "accuracy": {"score": 0, "details": "Error occurred"},
                    "relevance": {"score": 0, "details": "Error occurred"},
                    "completeness": {"score": 0, "details": "Error occurred"},
                    "hallucination": {"score": 0, "details": "Error occurred"}
                }
            
            # Record test case results
            test_result = {
                "id": test_case["id"],
                "query": test_case["query"],
                "category": test_case.get("category", "general"),
                "response": response,
                "processing_time": processing_time,
                "evaluation": evaluation_results,
                "error": error
            }
            
            results["test_cases"].append(test_result)
            
            if verbose:
                print(f"Processing time: {processing_time:.2f} seconds")
                if error:
                    print(f"Error: {error}")
                else:
                    print(f"Evaluation results:")
                    print(f"  Accuracy: {accuracy_score:.2f}")
                    print(f"  Relevance: {relevance_score:.2f}")
                    print(f"  Completeness: {completeness_score:.2f}")
                    print(f"  Hallucination: {hallucination_score:.2f}")
        
        # Calculate overall metrics
        if accuracy_scores:
            results["metrics"]["accuracy"] = sum(accuracy_scores) / len(accuracy_scores)
        if relevance_scores:
            results["metrics"]["relevance"] = sum(relevance_scores) / len(relevance_scores)
        if completeness_scores:
            results["metrics"]["completeness"] = sum(completeness_scores) / len(completeness_scores)
        if hallucination_scores:
            results["metrics"]["hallucination"] = sum(hallucination_scores) / len(hallucination_scores)
        
        # Calculate overall score
        if accuracy_scores:
            results["metrics"]["overall_score"] = (
                results["metrics"]["accuracy"] * 0.3 +
                results["metrics"]["relevance"] * 0.2 +
                results["metrics"]["completeness"] * 0.2 +
                results["metrics"]["hallucination"] * 0.3
            )
        
        # Calculate category-specific metrics
        categories = set(tc.get("category", "general") for tc in self.benchmark_data)
        category_metrics = {}
        
        for category in categories:
            category_results = [
                tc for tc in results["test_cases"] 
                if tc.get("category") == category
            ]
            
            if category_results:
                category_accuracy = sum(
                    r["evaluation"].get("accuracy", {}).get("score", 0) 
                    for r in category_results
                ) / len(category_results)
                
                category_metrics[category] = {
                    "count": len(category_results),
                    "accuracy": category_accuracy,
                }
        
        results["category_metrics"] = category_metrics
        
        # Save results if requested
        if self.save_results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = self.results_dir / f"benchmark_results_{timestamp}.json"
            
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved benchmark results to {results_file}")
        
        if verbose:
            print("\nOverall Benchmark Results:")
            print(f"  Accuracy: {results['metrics'].get('accuracy', 0):.4f}")
            print(f"  Relevance: {results['metrics'].get('relevance', 0):.4f}")
            print(f"  Completeness: {results['metrics'].get('completeness', 0):.4f}")
            print(f"  Hallucination: {results['metrics'].get('hallucination', 0):.4f}")
            print(f"  Overall Score: {results['metrics'].get('overall_score', 0):.4f}")
            
            print("\nCategory-Specific Results:")
            for category, metrics in results["category_metrics"].items():
                print(f"  {category} ({metrics['count']} test cases): {metrics['accuracy']:.4f}")
        
        logger.info("Benchmark evaluation completed")
        return results
    
    def create_custom_benchmark(
        self,
        test_cases: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Create a custom benchmark dataset.
        
        Args:
            test_cases: List of test cases for the benchmark.
            output_path: Path to save the benchmark dataset.
            
        Returns:
            Path to the saved benchmark dataset.
        """
        if output_path:
            save_path = Path(output_path)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = self.benchmark_data_path.parent / f"custom_benchmark_{timestamp}.json"
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the benchmark dataset
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(test_cases, f, indent=2)
        
        logger.info(f"Created custom benchmark dataset at {save_path}")
        return save_path
    
    def compare_benchmark_results(
        self,
        results_paths: List[Union[str, Path]],
        output_path: Optional[Union[str, Path]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Compare multiple benchmark results.
        
        Args:
            results_paths: List of paths to benchmark result files.
            output_path: Path to save the comparison results.
            verbose: Whether to print detailed results.
            
        Returns:
            Dictionary of comparison results.
        """
        # Load results
        results_list = []
        for path in results_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                results["file_path"] = str(path)
                results_list.append(results)
            except Exception as e:
                logger.error(f"Error loading benchmark results from {path}: {str(e)}")
        
        if not results_list:
            logger.error("No valid benchmark results to compare")
            return {}
        
        # Create comparison results
        comparison = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_results": len(results_list),
            "results": [
                {
                    "file_path": r["file_path"],
                    "timestamp": r.get("timestamp", "Unknown"),
                    "overall_score": r.get("metrics", {}).get("overall_score", 0),
                    "metrics": r.get("metrics", {})
                }
                for r in results_list
            ],
            "metric_comparison": {}
        }
        
        # Compare metrics
        metrics = ["accuracy", "relevance", "completeness", "hallucination", "overall_score"]
        for metric in metrics:
            values = [r.get("metrics", {}).get(metric, 0) for r in results_list]
            if values:
                comparison["metric_comparison"][metric] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "values": values
                }
        
        # Compare test case results
        test_case_ids = set()
        for result in results_list:
            test_case_ids.update(tc["id"] for tc in result.get("test_cases", []))
        
        test_case_comparison = {}
        for test_id in test_case_ids:
            test_case_comparison[test_id] = []
            for result in results_list:
                test_cases = result.get("test_cases", [])
                matching_cases = [tc for tc in test_cases if tc["id"] == test_id]
                if matching_cases:
                    test_case = matching_cases[0]
                    test_case_comparison[test_id].append({
                        "file_path": result["file_path"],
                        "accuracy": test_case.get("evaluation", {}).get("accuracy", {}).get("score", 0),
                        "processing_time": test_case.get("processing_time", 0)
                    })
        
        comparison["test_case_comparison"] = test_case_comparison
        
        # Save comparison if requested
        if output_path:
            save_path = Path(output_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2)
            
            logger.info(f"Saved comparison results to {save_path}")
        
        # Print verbose output if requested
        if verbose:
            print("\nBenchmark Comparison Results:")
            print(f"Comparing {len(results_list)} benchmark results")
            
            print("\nOverall Scores:")
            for i, result in enumerate(comparison["results"]):
                print(f"  Result {i+1} ({result['timestamp']}): {result['overall_score']:.4f}")
            
            print("\nMetric Comparison:")
            for metric, data in comparison["metric_comparison"].items():
                print(f"  {metric}:")
                print(f"    Min: {data['min']:.4f}")
                print(f"    Max: {data['max']:.4f}")
                print(f"    Avg: {data['avg']:.4f}")
        
        return comparison