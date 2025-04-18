"""
Search tool implementation for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool, ToolException
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

class SearchInput(BaseModel):
    """Input schema for the search tool."""
    query: str = Field(..., description="The search query to execute")
    num_results: int = Field(5, description="Number of search results to return")

class SearchTool(BaseTool):
    """
    Tool for performing web searches to retrieve up-to-date information.
    
    This tool uses DuckDuckGo Search to get recent information that might not
    be available in the BCG Sustainability Reports.
    """
    
    name = "web_search"
    description = "Search the web for recent or additional information not contained in the BCG reports"
    args_schema = SearchInput
    
    def _run(self, query: str, num_results: int = 5) -> str:
        """
        Run the search tool.
        
        Args:
            query: Search query.
            num_results: Number of results to return.
            
        Returns:
            Search results as a formatted string.
        """
        logger.info(f"Executing web search for query: '{query}'")
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            
            if not results:
                return "No search results found."
            
            # Format the results
            formatted_results = "Search Results:\n\n"
            
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                snippet = result.get("body", "No snippet")
                url = result.get("href", "No URL")
                
                formatted_results += f"{i}. {title}\n"
                formatted_results += f"   {snippet}\n"
                formatted_results += f"   Source: {url}\n\n"
            
            logger.info(f"Found {len(results)} search results for query: '{query}'")
            return formatted_results
        except Exception as e:
            error_message = f"Error executing search: {str(e)}"
            logger.error(error_message)
            raise ToolException(error_message)