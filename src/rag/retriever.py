"""
Retriever component for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

from src.embeddings.embeddings_manager import EmbeddingsManager

logger = logging.getLogger(__name__)

class EnhancedRetriever(BaseRetriever):
    """
    Enhanced retriever for retrieving relevant document chunks.
    
    This retriever extends the BaseRetriever with additional features like
    re-ranking, diversity optimization, and metadata filtering.
    """
    
    def __init__(
        self,
        embeddings_manager: EmbeddingsManager,
        search_kwargs: Optional[Dict[str, Any]] = None,
        use_mmr: bool = True,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ):
        """
        Initialize the enhanced retriever.
        
        Args:
            embeddings_manager: The embeddings manager instance.
            search_kwargs: Optional search parameters.
            use_mmr: Whether to use Maximal Marginal Relevance for diversity.
            fetch_k: Number of documents to fetch before reranking.
            lambda_mult: MMR lambda parameter (0-1), higher values prioritize relevance.
        """
        super().__init__()
        self.embeddings_manager = embeddings_manager
        self.search_kwargs = search_kwargs or {}
        self.use_mmr = use_mmr
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        
        logger.info(f"Enhanced retriever initialized with use_mmr={use_mmr}, fetch_k={fetch_k}")
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get documents relevant to the query.
        
        Args:
            query: The search query.
            run_manager: Callback manager for the retriever run.
            
        Returns:
            List of relevant Document objects.
        """
        logger.info(f"Retrieving documents for query: '{query}'")
        
        try:
            # Expand search_kwargs with instance attributes
            search_kwargs = {
                **self.search_kwargs,
                "use_mmr": self.use_mmr,
                "fetch_k": self.fetch_k,
                "lambda_mult": self.lambda_mult,
            }
            
            # If k is not explicitly set, use a default
            if "k" not in search_kwargs:
                search_kwargs["k"] = 4
            
            # Retrieve documents using embeddings manager
            documents = self.embeddings_manager.search(query, **search_kwargs)
            
            logger.info(f"Retrieved {len(documents)} documents for query: '{query}'")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise

class MultiQueryRetriever(BaseRetriever):
    """
    Multi-query retriever that generates multiple query variations to improve recall.
    
    This retriever uses an LLM to generate variations of the original query,
    performs retrieval for each query, and then combines and deduplicates the results.
    """
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm,
        num_queries: int = 3,
        deduplicate: bool = True,
    ):
        """
        Initialize the multi-query retriever.
        
        Args:
            base_retriever: The base retriever to use for individual queries.
            llm: Language model to generate query variations.
            num_queries: Number of query variations to generate.
            deduplicate: Whether to deduplicate results.
        """
        super().__init__()
        self.base_retriever = base_retriever
        self.llm = llm
        self.num_queries = num_queries
        self.deduplicate = deduplicate
        
        logger.info(f"Multi-query retriever initialized with num_queries={num_queries}")
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """
        Generate variations of the input query using an LLM.
        
        Args:
            query: The original query.
            
        Returns:
            List of query variations.
        """
        prompt = f"""
        You are an AI assistant helping to improve search results. Given a query, 
        your task is to generate {self.num_queries} different versions of the query 
        to maximize information retrieval. Each query should emphasize different aspects 
        of the original query.
        
        Original query: {query}
        
        Please provide {self.num_queries} different versions of the query.
        Your response should only include the queries, each on a new line.
        """
        
        try:
            response = self.llm.invoke(prompt)
            
            # Parse response to extract query variations
            variations = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Ensure we have the requested number of variations
            variations = variations[:self.num_queries]
            
            # Add the original query if not already included
            if query not in variations:
                variations.append(query)
            
            logger.info(f"Generated {len(variations)} query variations for: '{query}'")
            return variations
        except Exception as e:
            logger.error(f"Error generating query variations: {str(e)}")
            # Fall back to the original query
            return [query]
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get documents relevant to the query by combining results from multiple queries.
        
        Args:
            query: The search query.
            run_manager: Callback manager for the retriever run.
            
        Returns:
            List of relevant Document objects.
        """
        logger.info(f"Multi-query retrieval for: '{query}'")
        
        try:
            # Generate query variations
            queries = self._generate_query_variations(query)
            
            # Retrieve documents for each query
            all_docs = []
            for i, q in enumerate(queries):
                docs = self.base_retriever.get_relevant_documents(
                    q, 
                    callbacks=run_manager.get_child(f"query_{i}")
                )
                all_docs.extend(docs)
            
            # Deduplicate results if required
            if self.deduplicate:
                deduplicated_docs = []
                seen_contents = set()
                
                for doc in all_docs:
                    content = doc.page_content
                    if content not in seen_contents:
                        seen_contents.add(content)
                        deduplicated_docs.append(doc)
                
                all_docs = deduplicated_docs
            
            logger.info(f"Retrieved {len(all_docs)} documents across {len(queries)} queries")
            return all_docs
        except Exception as e:
            logger.error(f"Error in multi-query retrieval: {str(e)}")
            # Fall back to base retriever
            return self.base_retriever.get_relevant_documents(
                query, 
                callbacks=run_manager.get_child("fallback")
            )

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines semantic and keyword search.
    
    This retriever combines results from vector-based semantic search and
    traditional keyword-based search to improve both recall and precision.
    """
    
    def __init__(
        self,
        semantic_retriever: BaseRetriever,
        keyword_retriever: Optional[BaseRetriever] = None,
        semantic_weight: float = 0.7,
        merge_method: str = "interleave",
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            semantic_retriever: Retriever for semantic search.
            keyword_retriever: Retriever for keyword search (optional).
            semantic_weight: Weight given to semantic search results (0-1).
            merge_method: Method to merge results ("interleave" or "weighted").
        """
        super().__init__()
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.semantic_weight = semantic_weight
        self.merge_method = merge_method
        
        logger.info(f"Hybrid retriever initialized with merge_method={merge_method}")
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get documents relevant to the query using hybrid retrieval.
        
        Args:
            query: The search query.
            run_manager: Callback manager for the retriever run.
            
        Returns:
            List of relevant Document objects.
        """
        logger.info(f"Hybrid retrieval for: '{query}'")
        
        try:
            # Get semantic search results
            semantic_docs = self.semantic_retriever.get_relevant_documents(
                query, 
                callbacks=run_manager.get_child("semantic")
            )
            
            # If no keyword retriever is provided, return only semantic results
            if not self.keyword_retriever:
                return semantic_docs
            
            # Get keyword search results
            keyword_docs = self.keyword_retriever.get_relevant_documents(
                query, 
                callbacks=run_manager.get_child("keyword")
            )
            
            # Merge results based on the specified method
            if self.merge_method == "interleave":
                # Interleave results from both retrievers
                merged_docs = []
                for i in range(max(len(semantic_docs), len(keyword_docs))):
                    if i < len(semantic_docs):
                        merged_docs.append(semantic_docs[i])
                    if i < len(keyword_docs):
                        merged_docs.append(keyword_docs[i])
                
                # Deduplicate by content
                deduplicated_docs = []
                seen_contents = set()
                
                for doc in merged_docs:
                    content = doc.page_content
                    if content not in seen_contents:
                        seen_contents.add(content)
                        deduplicated_docs.append(doc)
                
                logger.info(f"Hybrid retrieval returned {len(deduplicated_docs)} documents")
                return deduplicated_docs
            
            elif self.merge_method == "weighted":
                # Combine results with weighted scores (not fully implemented in this version)
                # For a full implementation, we would need score information for each document
                # For now, we use semantic_weight to determine how many documents to take from each source
                semantic_count = int(self.semantic_weight * (len(semantic_docs) + len(keyword_docs)))
                keyword_count = len(semantic_docs) + len(keyword_docs) - semantic_count
                
                merged_docs = semantic_docs[:semantic_count] + keyword_docs[:keyword_count]
                
                # Deduplicate
                deduplicated_docs = []
                seen_contents = set()
                
                for doc in merged_docs:
                    content = doc.page_content
                    if content not in seen_contents:
                        seen_contents.add(content)
                        deduplicated_docs.append(doc)
                
                logger.info(f"Hybrid retrieval returned {len(deduplicated_docs)} documents")
                return deduplicated_docs
            
            else:
                # Default to returning semantic results
                logger.warning(f"Unknown merge method: {self.merge_method}, defaulting to semantic results")
                return semantic_docs
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            # Fall back to semantic retriever
            return self.semantic_retriever.get_relevant_documents(
                query, 
                callbacks=run_manager.get_child("fallback")
            )