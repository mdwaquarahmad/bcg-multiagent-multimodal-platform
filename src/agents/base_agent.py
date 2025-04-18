"""
Base agent implementation for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
from typing import Dict, List, Optional, Any, Union
import uuid

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base class for all agents in the BCG Multi-Agent & Multimodal AI Platform.
    
    This class provides the foundation for specialized agents, including
    common methods for initialization, message handling, and tool execution.
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        llm: BaseLanguageModel,
        system_prompt: str,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Name of the agent.
            role: Role/description of the agent.
            llm: Language model to use for the agent.
            system_prompt: System prompt that defines the agent's behavior.
            tools: Optional list of tools the agent can use.
            verbose: Whether to log verbose information.
        """
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.verbose = verbose
        self.memory = []  # List to store conversation history
        
        logger.info(f"Initialized {self.name} agent (ID: {self.agent_id})")
        
        if self.verbose:
            logger.info(f"System prompt: {self.system_prompt}")
            if self.tools:
                logger.info(f"Tools: {[tool.name for tool in self.tools]}")
    
    def add_message_to_memory(self, message: BaseMessage) -> None:
        """
        Add a message to the agent's memory.
        
        Args:
            message: Message to add to memory.
        """
        self.memory.append(message)
    
    def get_memory(self) -> List[BaseMessage]:
        """
        Get the agent's memory.
        
        Returns:
            List of messages in the agent's memory.
        """
        return self.memory
    
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self.memory = []
    
    def format_tool_descriptions(self) -> str:
        """
        Format tool descriptions for the system prompt.
        
        Returns:
            Formatted string of tool descriptions.
        """
        if not self.tools:
            return ""
        
        tool_descriptions = "You have access to the following tools:\n\n"
        
        for tool in self.tools:
            tool_descriptions += f"Tool: {tool.name}\n"
            tool_descriptions += f"Description: {tool.description}\n"
            if hasattr(tool, 'args_schema'):
                tool_descriptions += f"Args: {str(tool.args_schema)}\n"
            tool_descriptions += "\n"
        
        return tool_descriptions
    
    def format_messages(self, human_input: str) -> List[BaseMessage]:
        """
        Format messages for the LLM.
        
        Args:
            human_input: Human input message.
            
        Returns:
            List of formatted messages.
        """
        # Start with the system message that defines the agent
        messages = [
            SystemMessage(content=self.get_full_system_prompt())
        ]
        
        # Add relevant memory messages
        messages.extend(self.memory)
        
        # Add the human input
        messages.append(HumanMessage(content=human_input))
        
        return messages
    
    def get_full_system_prompt(self) -> str:
        """
        Get the full system prompt including tool descriptions.
        
        Returns:
            Complete system prompt string.
        """
        tool_descriptions = self.format_tool_descriptions()
        
        full_prompt = f"{self.system_prompt}\n\n"
        
        if tool_descriptions:
            full_prompt += f"{tool_descriptions}\n\n"
            
            # Add instructions for using tools
            full_prompt += """To use a tool, please use the following format:
Thought: I need to use a tool to help answer the question.
Action: tool_name
Action Input: {
"param1": "value1",
"param2": "value2"
}

After using a tool, I'll receive the result and can use it to inform my response or use another tool.
"""
        
        return full_prompt
    
    def run(self, human_input: str) -> str:
        """
        Run the agent on an input.
        
        Args:
            human_input: Human input message.
            
        Returns:
            Agent's response.
        """
        # Format messages
        messages = self.format_messages(human_input)
        
        # Log the input if verbose
        if self.verbose:
            logger.info(f"{self.name} received input: {human_input}")
        
        # Generate response
        ai_message = self.llm.invoke(messages)
        
        # Add the input and response to memory
        self.add_message_to_memory(HumanMessage(content=human_input))
        self.add_message_to_memory(ai_message)
        
        # Process for tool use
        response = ai_message.content
        tool_use = self._extract_tool_use(response)
        
        if tool_use:
            # Agent wants to use a tool
            tool_name, tool_input = tool_use
            
            # Find the requested tool
            selected_tool = None
            for tool in self.tools:
                if tool.name == tool_name:
                    selected_tool = tool
                    break
            
            if selected_tool:
                # Execute the tool
                try:
                    tool_result = selected_tool.invoke(tool_input)
                    tool_message = f"Tool: {tool_name}\nResult: {tool_result}"
                    
                    # Add tool result to memory
                    self.add_message_to_memory(AIMessage(content=response))
                    self.add_message_to_memory(HumanMessage(content=tool_message))
                    
                    # Generate final response with tool result
                    final_messages = self.format_messages(tool_message)
                    final_response = self.llm.invoke(final_messages)
                    
                    # Add final response to memory
                    self.add_message_to_memory(final_response)
                    
                    if self.verbose:
                        logger.info(f"{self.name} used tool {tool_name}: {tool_input}")
                        logger.info(f"Tool result: {tool_result}")
                        logger.info(f"{self.name} final response: {final_response.content}")
                    
                    return final_response.content
                except Exception as e:
                    error_message = f"Error executing tool {tool_name}: {str(e)}"
                    logger.error(error_message)
                    
                    # Add error message to memory
                    self.add_message_to_memory(AIMessage(content=response))
                    self.add_message_to_memory(HumanMessage(content=error_message))
                    
                    # Generate new response with error information
                    error_messages = self.format_messages(error_message)
                    error_response = self.llm.invoke(error_messages)
                    
                    # Add error response to memory
                    self.add_message_to_memory(error_response)
                    
                    return error_response.content
            else:
                # Tool not found
                error_message = f"Tool '{tool_name}' not found."
                logger.error(error_message)
                
                # Add error message to memory
                self.add_message_to_memory(AIMessage(content=response))
                self.add_message_to_memory(HumanMessage(content=error_message))
                
                # Generate new response with error information
                error_messages = self.format_messages(error_message)
                error_response = self.llm.invoke(error_messages)
                
                # Add error response to memory
                self.add_message_to_memory(error_response)
                
                return error_response.content
        
        # No tool use, return the response directly
        if self.verbose:
            logger.info(f"{self.name} response: {response}")
        
        return response
    
    def _extract_tool_use(self, response: str) -> Optional[tuple]:
        """
        Extract tool use from the agent's response.
        
        Args:
            response: Agent's response.
            
        Returns:
            Tuple of (tool_name, tool_input) if a tool is being used, None otherwise.
        """
        if not self.tools:
            return None
        
        # Check for tool usage format
        if "Action:" in response and "Action Input:" in response:
            try:
                # Extract the tool name
                action_start = response.find("Action:") + len("Action:")
                action_end = response.find("\n", action_start)
                tool_name = response[action_start:action_end].strip()
                
                # Extract the tool input
                input_start = response.find("Action Input:") + len("Action Input:")
                input_end = response.find("```", input_start) if "```" in response[input_start:] else len(response)
                tool_input_str = response[input_start:input_end].strip()
                
                # Parse the tool input
                import json
                # Handle both JSON format and plain text
                try:
                    tool_input = json.loads(tool_input_str)
                except json.JSONDecodeError:
                    # If not valid JSON, use as raw string
                    tool_input = tool_input_str
                
                return (tool_name, tool_input)
            except Exception as e:
                logger.error(f"Error extracting tool use: {str(e)}")
                return None
        
        return None