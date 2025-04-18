"""
RAG pipeline for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate

from src.embeddings.embeddings_manager import EmbeddingsManager
from src.rag.retriever import EnhancedRetriever, MultiQueryRetriever, HybridRetriever
from src.rag.prompt_builder import PromptBuilder
from src.rag.generator import ResponseGenerator

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Data class for storing the RAG pipeline response."""
    query: str
    response: str
    source_documents: List[Document]
    prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class RAGPipeline:
    """
    End-to-end RAG pipeline for BCG Multi-Agent & Multimodal AI Platform.
    
    This class orchestrates the entire RAG pipeline, including retrieval,
    prompt building, and response generation.
    """
    
    def __init__(
        self,
        embeddings_manager: EmbeddingsManager,
        response_generator: ResponseGenerator,
        retriever_type: str = "enhanced",
        use_multi_query: bool = False,
        include_sources: bool = True,
        max_sources: int = 4,
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embeddings_manager: The embeddings manager instance.
            response_generator: The response generator instance.
            retriever_type: Type of retriever to use ('enhanced', 'hybrid').
            use_multi_query: Whether to use multi-query retrieval.
            include_sources: Whether to include source citations in responses.
            max_sources: Maximum number of sources to retrieve.
        """
        self.embeddings_manager = embeddings_manager
        self.response_generator = response_generator
        self.retriever_type = retriever_type
        self.use_multi_query = use_multi_query
        self.include_sources = include_sources
        self.max_sources = max_sources
        
        # Initialize the prompt builder
        self.prompt_builder = PromptBuilder(
            include_source_documents=include_sources,
        )
        
        # Initialize the retriever
        self.retriever = self._create_retriever()
        
        logger.info(f"RAG pipeline initialized with {retriever_type} retriever")
    
    def _create_retriever(self) -> BaseRetriever:
        """
        Create the retriever based on the configured type.
        
        Returns:
            Configured retriever instance.
        """
        # Create the base retriever
        if self.retriever_type == "enhanced":
            base_retriever = EnhancedRetriever(
                embeddings_manager=self.embeddings_manager,
                search_kwargs={"k": self.max_sources},
                use_mmr=True,
            )
        elif self.retriever_type == "hybrid":
            # Hybrid retriever uses semantic and keyword search
            # Note: In a full implementation, we would create a keyword retriever
            # For now, we just use the enhanced retriever for both
            semantic_retriever = EnhancedRetriever(
                embeddings_manager=self.embeddings_manager,
                search_kwargs={"k": self.max_sources},
                use_mmr=True,
            )
            base_retriever = HybridRetriever(
                semantic_retriever=semantic_retriever,
            )
        else:
            # Default to enhanced retriever
            base_retriever = EnhancedRetriever(
                embeddings_manager=self.embeddings_manager,
                search_kwargs={"k": self.max_sources},
                use_mmr=True,
            )
        
        # Wrap with multi-query retriever if enabled
        if self.use_multi_query:
            return MultiQueryRetriever(
                base_retriever=base_retriever,
                llm=self.response_generator.get_llm(),
                num_queries=3,
            )
        
        return base_retriever
    
    def query(
        self,
        query: str,
        chat_history: Optional[List[Union[HumanMessage, AIMessage]]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: The user query.
            chat_history: Optional chat history for context.
            filter_criteria: Optional filter criteria for retrieval.
            
        Returns:
            RAGResponse containing the response and metadata.
        """
        logger.info(f"Processing query: '{query}'")
        
        try:
            # Update retriever filter if provided
            if filter_criteria:
                if hasattr(self.retriever, "search_kwargs"):
                    self.retriever.search_kwargs["filter"] = filter_criteria
                elif hasattr(self.retriever, "base_retriever") and hasattr(self.retriever.base_retriever, "search_kwargs"):
                    self.retriever.base_retriever.search_kwargs["filter"] = filter_criteria
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.get_relevant_documents(query)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: '{query}'")
            
            # Build the prompt
            prompt = self.prompt_builder.build_rag_prompt(
                query=query,
                documents=retrieved_docs,
                chat_history=chat_history,
            )
            
            # Generate the response
            response = self.response_generator.generate_response(
                prompt=prompt,
                chat_history=chat_history,
            )
            
            logger.info(f"Generated response for query: '{query}'")
            
            # Create and return the RAG response
            return RAGResponse(
                query=query,
                response=response,
                source_documents=retrieved_docs,
                prompt=str(prompt),
                metadata={
                    "num_source_documents": len(retrieved_docs),
                    "retriever_type": self.retriever_type,
                    "use_multi_query": self.use_multi_query,
                }
            )
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            # Return a fallback response
            return RAGResponse(
                query=query,
                response=f"I'm sorry, I encountered an error while processing your query: {str(e)}",
                source_documents=[],
                metadata={"error": str(e)},
            )
    
    def generate_summary(self, topic: Optional[str] = None) -> RAGResponse:
        """
        Generate a summary of the documents.
        
        Args:
            topic: Optional topic to focus the summary on.
            
        Returns:
            RAGResponse containing the summary.
        """
        logger.info(f"Generating summary{' on ' + topic if topic else ''}")
        
        try:
            # Retrieve documents
            if topic:
                # Focus on the specified topic
                query = f"Summarize information about {topic} in BCG sustainability reports"
                filter_criteria = None  # Filter could be added based on topic
            else:
                # General summary
                query = "Summarize the key points from BCG sustainability reports"
                filter_criteria = None
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.get_relevant_documents(query)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for summary")
            
            # Build the summary prompt
            prompt = self.prompt_builder.build_summary_prompt(documents=retrieved_docs)
            
            # Generate the summary
            summary = self.response_generator.generate_response(prompt=prompt)
            
            logger.info("Generated summary successfully")
            
            # Create and return the RAG response
            return RAGResponse(
                query=query,
                response=summary,
                source_documents=retrieved_docs,
                prompt=str(prompt),
                metadata={
                    "type": "summary",
                    "topic": topic,
                    "num_source_documents": len(retrieved_docs),
                }
            )
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Return a fallback response
            return RAGResponse(
                query=f"Summary{' on ' + topic if topic else ''}",
                response=f"I'm sorry, I encountered an error while generating the summary: {str(e)}",
                source_documents=[],
                metadata={"error": str(e)},
            )
    
    def compare_across_years(self, topic: str) -> RAGResponse:
        """
        Compare information on a topic across different years.
        
        Args:
            topic: Topic to compare.
            
        Returns:
            RAGResponse containing the comparison.
        """
        logger.info(f"Comparing topic '{topic}' across years")
        
        try:
            # Query to find relevant documents
            query = f"{topic} in BCG sustainability reports"
            
            # Retrieve relevant documents
            all_docs = self.retriever.get_relevant_documents(query)
            
            logger.info(f"Retrieved {len(all_docs)} documents for comparison")
            
            # Group documents by year based on metadata
            docs_by_year = {}
            for doc in all_docs:
                # Try to extract year from document metadata
                year = None
                if "filename" in doc.metadata:
                    filename = doc.metadata["filename"]
                    # Extract year from filename (assuming format like "BCG-2023-Annual-Sustainability-Report.pdf")
                    import re
                    year_match = re.search(r'20\d{2}', filename)
                    if year_match:
                        year = year_match.group(0)
                
                if not year:
                    # Try to extract from document content or other metadata
                    year = "Unknown"
                
                if year not in docs_by_year:
                    docs_by_year[year] = []
                
                docs_by_year[year].append(doc)
            
            # Build comparison prompt
            prompt = self.prompt_builder.build_comparison_prompt(
                topic=topic,
                documents_by_year=docs_by_year,
            )
            
            # Generate the comparison
            comparison = self.response_generator.generate_response(prompt=prompt)
            
            logger.info(f"Generated comparison for topic '{topic}'")
            
            # Create and return the RAG response
            return RAGResponse(
                query=f"Compare {topic} across years",
                response=comparison,
                source_documents=all_docs,
                prompt=str(prompt),
                metadata={
                    "type": "comparison",
                    "topic": topic,
                    "years": list(docs_by_year.keys()),
                    "num_source_documents": len(all_docs),
                }
            )
        except Exception as e:
            logger.error(f"Error comparing across years: {str(e)}")
            # Return a fallback response
            return RAGResponse(
                query=f"Compare {topic} across years",
                response=f"I'm sorry, I encountered an error while comparing the topic across years: {str(e)}",
                source_documents=[],
                metadata={"error": str(e)},
            )
    
    def extract_facts(self, fact_type: str = "metrics") -> RAGResponse:
        """
        Extract facts or metrics from the documents.
        
        Args:
            fact_type: Type of facts to extract ('metrics', 'commitments', etc.).
            
        Returns:
            RAGResponse containing the extracted facts.
        """
        logger.info(f"Extracting {fact_type} from documents")
        
        try:
            # Query to find relevant documents
            query = f"{fact_type} in BCG sustainability reports"
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.get_relevant_documents(query)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for fact extraction")
            
            # Build fact extraction prompt
            prompt = self.prompt_builder.build_fact_extraction_prompt(
                documents=retrieved_docs,
                fact_type=fact_type,
            )
            
            # Generate the facts extraction
            facts = self.response_generator.generate_response(prompt=prompt)
            
            logger.info(f"Extracted {fact_type} successfully")
            
            # Create and return the RAG response
            return RAGResponse(
                query=f"Extract {fact_type} from BCG sustainability reports",
                response=facts,
                source_documents=retrieved_docs,
                prompt=str(prompt),
                metadata={
                    "type": "fact_extraction",
                    "fact_type": fact_type,
                    "num_source_documents": len(retrieved_docs),
                }
            )
        except Exception as e:
            logger.error(f"Error extracting facts: {str(e)}")
            # Return a fallback response
            return RAGResponse(
                query=f"Extract {fact_type}",
                response=f"I'm sorry, I encountered an error while extracting {fact_type}: {str(e)}",
                source_documents=[],
                metadata={"error": str(e)},
            )