"""
Generator component for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain.callbacks.manager import CallbackManager
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Generates responses using LLMs based on prompts and contexts.
    
    This class handles the interaction with language models to generate
    responses for various use cases in the BCG Multi-Agent & Multimodal AI Platform.
    """
    
    def __init__(
        self,
        model_name: str = "gemma3:4b",
        model_type: str = "ollama",
        temperature: float = 0.2,
        streaming: bool = False,
        max_tokens: Optional[int] = 2000,
        api_key: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
    ):
        """
        Initialize the response generator.
        
        Args:
            model_name: Name of the language model to use.
            model_type: Type of language model ('ollama', 'openai').
            temperature: Sampling temperature for the model.
            streaming: Whether to use streaming for response generation.
            max_tokens: Maximum number of tokens to generate.
            api_key: API key for external models (required for 'openai').
            ollama_base_url: Base URL for Ollama API (optional).
        """
        self.model_name = model_name
        self.model_type = model_type.lower()
        self.temperature = temperature
        self.streaming = streaming
        
        # Set up callback manager for streaming if enabled
        self.callback_manager = None
        if streaming:
            self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # Initialize the language model based on the model type
        if self.model_type == "openai":
            # Use OpenAI model
            if not api_key and not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key is required for OpenAI models")
            
            logger.info(f"Initializing OpenAI model: {model_name}")
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                streaming=streaming,
                openai_api_key=api_key or os.getenv("OPENAI_API_KEY"),
                max_tokens=max_tokens,
                callback_manager=self.callback_manager,
            )
        elif self.model_type == "ollama":
            # Use Ollama model
            base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            logger.info(f"Initializing Ollama model: {model_name} at {base_url}")
            self.llm = Ollama(
                model=model_name,
                temperature=temperature,
                base_url=base_url,
                callback_manager=self.callback_manager,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Choose 'ollama' or 'openai'.")
    
    def generate_response(
        self,
        prompt: Union[str, ChatPromptTemplate],
        chat_history: Optional[List[Union[HumanMessage, AIMessage]]] = None,
    ) -> str:
        """
        Generate a response using the language model.
        
        Args:
            prompt: The prompt to generate a response for.
            chat_history: Optional chat history for context.
            
        Returns:
            The generated response as a string.
        """
        try:
            if isinstance(prompt, str):
                # Simple string prompt
                logger.info(f"Generating response for string prompt")
                response = self.llm.invoke(prompt)
            else:
                # ChatPromptTemplate with potential placeholders
                logger.info(f"Generating response for chat prompt template")
                
                # Prepare the input dictionary for the prompt
                prompt_args = {}
                if "chat_history" in prompt.input_variables and chat_history:
                    prompt_args["chat_history"] = chat_history
                
                # Create the chain and generate the response
                chain = prompt | self.llm
                response = chain.invoke(prompt_args)
            
            # Extract the response content
            if hasattr(response, "content"):
                return response.content
            else:
                return str(response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def get_llm(self) -> BaseChatModel:
        """
        Get the underlying language model.
        
        Returns:
            The language model instance.
        """
        return self.llm