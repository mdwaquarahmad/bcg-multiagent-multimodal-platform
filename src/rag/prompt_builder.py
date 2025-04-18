"""
Prompt builder for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
from typing import Dict, List, Optional, Union, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    Builds prompts for the BCG Multi-Agent & Multimodal AI Platform.
    
    This class creates structured prompts for various RAG use cases,
    incorporating retrieved context and user queries.
    """
    
    def __init__(
        self,
        use_structured_output: bool = False,
        include_source_documents: bool = True,
        max_context_length: int = 10000,
    ):
        """
        Initialize the prompt builder.
        
        Args:
            use_structured_output: Whether to request structured output from the LLM.
            include_source_documents: Whether to include source citations in the output.
            max_context_length: Maximum length of context to include in the prompt.
        """
        self.use_structured_output = use_structured_output
        self.include_source_documents = include_source_documents
        self.max_context_length = max_context_length
        
        logger.info(f"Prompt builder initialized with max_context_length={max_context_length}")
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """
        Prepare context from retrieved documents.
        
        Args:
            documents: Retrieved documents.
            
        Returns:
            Formatted context string.
        """
        if not documents:
            return "No relevant information found."
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            # Extract source information
            source_info = self._get_source_info(doc)
            
            # Format document with source info
            doc_text = f"[Document {i+1}] {source_info}\n{doc.page_content}"
            
            # Check if adding this document exceeds the maximum context length
            if current_length + len(doc_text) > self.max_context_length:
                # If we already have at least one document, stop adding more
                if context_parts:
                    break
                
                # If this is the first document, truncate it to fit
                truncated_text = doc_text[:self.max_context_length - 100] + "... [truncated]"
                context_parts.append(truncated_text)
                current_length = len(truncated_text)
            else:
                context_parts.append(doc_text)
                current_length += len(doc_text)
        
        return "\n\n".join(context_parts)
    
    def _get_source_info(self, doc: Document) -> str:
        """
        Extract source information from a document.
        
        Args:
            doc: Document to extract source info from.
            
        Returns:
            Formatted source information string.
        """
        metadata = doc.metadata or {}
        
        # Extract key information
        document_id = metadata.get("document_id", "Unknown")
        filename = metadata.get("filename", "Unknown")
        page = metadata.get("page", "")
        page_info = f", Page: {page}" if page else ""
        
        return f"Source: {filename} (ID: {document_id}{page_info})"
    
    def build_rag_prompt(
        self,
        query: str,
        documents: List[Document],
        chat_history: Optional[List[Union[HumanMessage, AIMessage]]] = None,
    ) -> ChatPromptTemplate:
        """
        Build a prompt for RAG.
        
        Args:
            query: User query.
            documents: Retrieved documents.
            chat_history: Optional chat history.
            
        Returns:
            ChatPromptTemplate for the RAG system.
        """
        logger.info(f"Building RAG prompt for query: '{query}'")
        
        # Prepare context from documents
        context = self._prepare_context(documents)
        
        # Define the system message
        system_template = """You are a helpful AI assistant specialized in analyzing BCG Sustainability Reports. 
You provide accurate, informative, and helpful responses based on the context provided.

Please follow these guidelines:
1. Base your answers only on the context provided. Do not use prior knowledge about BCG.
2. If the context doesn't contain the answer, say "I don't have enough information about that in the provided reports."
3. Be concise but comprehensive.
4. When referring to specific data or statements, cite the source document.
{structured_output_instruction}

Context:
{context}
"""
        
        # Add instruction for structured output if required
        structured_output_instruction = """
5. Format your response in markdown for readability.
6. For citations, use the format [Document X] at the end of the relevant sentence or paragraph.
""" if self.include_source_documents else ""
        
        # Create the system message
        system_message = SystemMessage(
            content=system_template.format(
                context=context,
                structured_output_instruction=structured_output_instruction,
            )
        )
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                system_message,
                # Include chat history if provided
                MessagesPlaceholder(variable_name="chat_history") if chat_history else None,
                # User query
                HumanMessage(content=query),
            ]
        )
        
        # Remove None values from the messages list
        prompt.messages = [msg for msg in prompt.messages if msg is not None]
        
        return prompt
    
    def build_query_analysis_prompt(self, query: str) -> ChatPromptTemplate:
        """
        Build a prompt for query analysis.
        
        Args:
            query: User query.
            
        Returns:
            ChatPromptTemplate for query analysis.
        """
        logger.info(f"Building query analysis prompt for: '{query}'")
        
        system_template = """You are an AI assistant that helps analyze user queries to improve search results.
Your task is to:
1. Identify the key concepts and entities in the query.
2. Determine the main intent of the query.
3. Suggest any additional relevant terms or concepts that might improve search results.

Please provide your analysis in a clear, structured format."""
        
        system_message = SystemMessage(content=system_template)
        
        prompt = ChatPromptTemplate.from_messages(
            [
                system_message,
                HumanMessage(content=f"Please analyze the following query: '{query}'"),
            ]
        )
        
        return prompt
    
    def build_summary_prompt(self, documents: List[Document]) -> ChatPromptTemplate:
        """
        Build a prompt for summarizing documents.
        
        Args:
            documents: Documents to summarize.
            
        Returns:
            ChatPromptTemplate for summarization.
        """
        logger.info(f"Building summary prompt for {len(documents)} documents")
        
        # Prepare context from documents
        context = self._prepare_context(documents)
        
        system_template = """You are an AI assistant that specializes in summarizing BCG sustainability reports.
Your task is to provide a comprehensive yet concise summary of the provided content.

Please follow these guidelines:
1. Focus on the key themes, initiatives, and metrics from the content.
2. Organize the summary in a logical structure with clear sections.
3. Highlight significant achievements, goals, and changes compared to previous years if mentioned.
4. Use bullet points where appropriate for clarity.
5. Format your response in markdown for readability.

Content to summarize:
{context}"""
        
        system_message = SystemMessage(content=system_template.format(context=context))
        
        prompt = ChatPromptTemplate.from_messages(
            [
                system_message,
                HumanMessage(content="Please provide a comprehensive summary of these BCG sustainability reports."),
            ]
        )
        
        return prompt
    
    def build_comparison_prompt(
        self,
        topic: str,
        documents_by_year: Dict[str, List[Document]],
    ) -> ChatPromptTemplate:
        """
        Build a prompt for comparing documents from different years.
        
        Args:
            topic: Topic to compare.
            documents_by_year: Dictionary mapping years to lists of documents.
            
        Returns:
            ChatPromptTemplate for comparison.
        """
        logger.info(f"Building comparison prompt for topic: '{topic}'")
        
        # Prepare context from each year's documents
        contexts_by_year = {}
        for year, docs in documents_by_year.items():
            contexts_by_year[year] = self._prepare_context(docs)
        
        # Combine all contexts with clear separation
        combined_context = ""
        for year, context in contexts_by_year.items():
            combined_context += f"\n\n### CONTENT FROM {year} REPORT ###\n\n{context}"
        
        system_template = """You are an AI assistant that specializes in analyzing and comparing BCG sustainability reports across different years.
Your task is to compare how the topic of '{topic}' has evolved over time based on the provided content from different years.

Please follow these guidelines:
1. Identify key similarities and differences in how the topic is addressed across the reports.
2. Note any evolution in approach, commitment, metrics, or achievements related to the topic.
3. Highlight significant changes or new initiatives introduced in more recent reports.
4. Organize your comparison chronologically, showing how BCG's approach has developed over time.
5. Format your response in markdown for readability, using tables for direct comparisons where appropriate.

Content to compare:
{context}"""
        
        system_message = SystemMessage(
            content=system_template.format(
                topic=topic,
                context=combined_context,
            )
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                system_message,
                HumanMessage(content=f"Please compare how BCG's approach to '{topic}' has evolved over the years in these sustainability reports."),
            ]
        )
        
        return prompt
    
    def build_fact_extraction_prompt(
        self,
        documents: List[Document],
        fact_type: str = "metrics",
    ) -> ChatPromptTemplate:
        """
        Build a prompt for extracting facts or metrics from documents.
        
        Args:
            documents: Documents to extract facts from.
            fact_type: Type of facts to extract ('metrics', 'commitments', etc.).
            
        Returns:
            ChatPromptTemplate for fact extraction.
        """
        logger.info(f"Building fact extraction prompt for {fact_type}")
        
        # Prepare context from documents
        context = self._prepare_context(documents)
        
        system_template = """You are an AI assistant that specializes in extracting {fact_type} from BCG sustainability reports.
Your task is to identify and extract all {fact_type} mentioned in the provided content.

Please follow these guidelines:
1. Focus only on clearly stated {fact_type} with specific details.
2. For each {fact_type}, include the relevant value, year, and context.
3. Organize the {fact_type} by category (e.g., environmental, social, governance).
4. If numerical, note the units and any comparison to previous periods.
5. Cite the source document for each {fact_type}.
6. Format your response as a structured list in markdown.

Content to analyze:
{context}"""
        
        system_message = SystemMessage(
            content=system_template.format(
                fact_type=fact_type,
                context=context,
            )
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                system_message,
                HumanMessage(content=f"Please extract all {fact_type} from these BCG sustainability reports."),
            ]
        )
        
        return prompt