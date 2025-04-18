"""
Agent graph implementation for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple, Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.agents.researcher_agent import ResearcherAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.visual_analyst_agent import VisualAnalystAgent
from src.agents.strategist_agent import StrategistAgent
from src.agents.critic_agent import CriticAgent
from src.agents.tools.search_tool import SearchTool
from src.agents.tools.python_tool import PythonTool
from src.agents.tools.chart_analysis_tool import ChartAnalysisTool

logger = logging.getLogger(__name__)

# Define the state schema
class AgentState(TypedDict):
    """State for the agent graph."""
    messages: List[BaseMessage]
    query: str
    context: Dict[str, Any]
    research_results: Optional[Dict[str, Any]]
    analysis_results: Optional[Dict[str, Any]]
    visual_analysis_results: Optional[Dict[str, Any]]
    strategy_results: Optional[Dict[str, Any]]
    critique_results: Optional[Dict[str, Any]]
    final_response: Optional[str]
    current_agent: str
    next_agent: Optional[str]

class BCGMultiAgentSystem:
    """
    Multi-agent system for BCG Multi-Agent & Multimodal AI Platform.
    
    This class orchestrates multiple specialized agents using LangGraph to create
    a collaborative multi-agent system that analyzes BCG Sustainability Reports.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        visuals_directory: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the multi-agent system.
        
        Args:
            llm: Language model to use for all agents.
            visuals_directory: Directory containing visual elements to analyze.
            verbose: Whether to log verbose information.
        """
        self.llm = llm
        self.visuals_directory = visuals_directory
        self.verbose = verbose
        
        # Initialize agents
        self.researcher_agent = ResearcherAgent(
            llm=self.llm,
            verbose=self.verbose,
        )
        
        self.analyst_agent = AnalystAgent(
            llm=self.llm,
            verbose=self.verbose,
        )
        
        self.visual_analyst_agent = VisualAnalystAgent(
            llm=self.llm,
            visuals_directory=self.visuals_directory,
            verbose=self.verbose,
        )
        
        self.strategist_agent = StrategistAgent(
            llm=self.llm,
            verbose=self.verbose,
        )
        
        self.critic_agent = CriticAgent(
            llm=self.llm,
            verbose=self.verbose,
        )
        
        # Build the agent graph
        self.workflow = self._build_graph()
        
        logger.info("BCG multi-agent system initialized")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the agent graph using LangGraph.
        
        Returns:
            StateGraph for the multi-agent workflow.
        """
        # Create the graph
        graph = StateGraph(AgentState)
        
        # Add nodes for each agent
        graph.add_node("coordinator", self._coordinator_node)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("visual_analyst", self._visual_analyst_node)
        graph.add_node("strategist", self._strategist_node)
        graph.add_node("critic", self._critic_node)
        graph.add_node("finalizer", self._finalizer_node)
        
        # Define the edges
        # Start with the coordinator
        graph.add_edge("coordinator", self._route_after_coordinator)
        
        # Connect researcher to analyst or visual analyst
        graph.add_conditional_edges(
            "researcher",
            self._route_after_researcher,
            {
                "analyst": "analyst",
                "visual_analyst": "visual_analyst",
                "strategist": "strategist",
            }
        )
        
        # Connect analyst to strategist or visual analyst
        graph.add_conditional_edges(
            "analyst",
            self._route_after_analyst,
            {
                "visual_analyst": "visual_analyst",
                "strategist": "strategist",
            }
        )
        
        # Connect visual analyst to strategist
        graph.add_edge("visual_analyst", "strategist")
        
        # Connect strategist to critic
        graph.add_edge("strategist", "critic")
        
        # Connect critic to finalizer or back to appropriate agent
        graph.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "visual_analyst": "visual_analyst",
                "strategist": "strategist",
                "finalizer": "finalizer",
            }
        )
        
        # Connect finalizer to END
        graph.add_edge("finalizer", END)
        
        # Set the entry point
        graph.set_entry_point("coordinator")
        
        return graph.compile()
    
    def _coordinator_node(self, state: AgentState) -> AgentState:
        """
        Coordinator node that routes the initial query.
        
        Args:
            state: Current state.
            
        Returns:
            Updated state.
        """
        query = state["query"]
        
        # Determine the query type and initial routing
        coordinator_prompt = f"""As the coordinator for the BCG multi-agent system, your task is to analyze this query and determine which agent should handle it first:

Query: {query}

Please analyze what this query is asking for and determine which of our specialized agents should tackle it first:

1. Researcher Agent: For finding specific information and gathering facts
2. Analyst Agent: For data analysis and pattern recognition
3. Visual Analyst Agent: For analyzing charts, graphs, and visual elements
4. Other (specify): If a different approach is needed

Consider what the query is primarily asking for. Is it seeking:
- Specific information that requires research?
- Analysis of data or trends?
- Interpretation of visual elements?
- Something else entirely?

Respond with just the name of the appropriate agent to start with: "researcher", "analyst", "visual_analyst", or explain briefly if it requires a different approach."""
        
        # Use the LLM to route
        response = self.llm.invoke(coordinator_prompt)
        
        # Extract the agent name from the response
        agent_name = response.content.strip().lower()
        
        # Default to researcher if unclear
        if "researcher" in agent_name:
            next_agent = "researcher"
        elif "analyst" in agent_name:
            next_agent = "analyst"
        elif "visual" in agent_name:
            next_agent = "visual_analyst"
        else:
            # Default to researcher if unclear
            next_agent = "researcher"
        
        # Update the state
        state["next_agent"] = next_agent
        state["current_agent"] = "coordinator"
        state["messages"].append(HumanMessage(content=query))
        
        if self.verbose:
            logger.info(f"Coordinator routing query to {next_agent}")
        
        return state
    
    def _researcher_node(self, state: AgentState) -> AgentState:
        """
        Researcher node that gathers information.
        
        Args:
            state: Current state.
            
        Returns:
            Updated state.
        """
        query = state["query"]
        
        # Run the researcher agent
        research_prompt = f"""I need you to research the following query about BCG's sustainability efforts:

{query}

Please gather comprehensive information from BCG's sustainability reports and any other relevant sources. Focus on collecting factual data, key statistics, and important context.

Use your search tool if you need additional information beyond what's available in the BCG reports."""
        
        research_results = self.researcher_agent.run(research_prompt)
        
        # Update the state
        state["research_results"] = {
            "content": research_results,
            "timestamp": str(uuid.uuid4()),
        }
        state["current_agent"] = "researcher"
        state["messages"].append(AIMessage(content=research_results))
        
        if self.verbose:
            logger.info(f"Researcher completed research: {len(research_results)} chars")
        
        return state
    
    def _analyst_node(self, state: AgentState) -> AgentState:
        """
        Analyst node that performs data analysis.
        
        Args:
            state: Current state.
            
        Returns:
            Updated state.
        """
        query = state["query"]
        research_results = state.get("research_results", {}).get("content", "")
        
        # Run the analyst agent
        analysis_prompt = f"""I need you to analyze the following information about BCG's sustainability efforts:

Research Information:
{research_results}

Original Query: {query}

Please analyze this information to identify trends, patterns, and insights. Use your python_executor tool if you need to perform any calculations or data analysis.

Focus on:
1. Key metrics and their evolution over time
2. Significant patterns or correlations
3. Comparative analysis across different areas or years
4. Data-backed insights relevant to the query"""
        
        analysis_results = self.analyst_agent.run(analysis_prompt)
        
        # Update the state
        state["analysis_results"] = {
            "content": analysis_results,
            "timestamp": str(uuid.uuid4()),
        }
        state["current_agent"] = "analyst"
        state["messages"].append(AIMessage(content=analysis_results))
        
        if self.verbose:
            logger.info(f"Analyst completed analysis: {len(analysis_results)} chars")
        
        return state
    
    def _visual_analyst_node(self, state: AgentState) -> AgentState:
        """
        Visual analyst node that analyzes visual elements.
        
        Args:
            state: Current state.
            
        Returns:
            Updated state.
        """
        query = state["query"]
        research_results = state.get("research_results", {}).get("content", "")
        
        # Run the visual analyst agent
        visual_prompt = f"""I need you to analyze visual elements related to BCG's sustainability efforts that are relevant to this query:

Query: {query}

Context from Research:
{research_results}

Please identify and analyze any relevant charts, graphs, or visual elements that would help answer this query. Use your chart_analyzer tool to examine specific visuals.

Focus on:
1. Identifying relevant visual elements
2. Extracting data and insights from these visuals
3. Interpreting what these visuals reveal about BCG's sustainability efforts
4. Explaining how the visual data relates to the query"""
        
        visual_analysis_results = self.visual_analyst_agent.run(visual_prompt)
        
        # Update the state
        state["visual_analysis_results"] = {
            "content": visual_analysis_results,
            "timestamp": str(uuid.uuid4()),
        }
        state["current_agent"] = "visual_analyst"
        state["messages"].append(AIMessage(content=visual_analysis_results))
        
        if self.verbose:
            logger.info(f"Visual Analyst completed analysis: {len(visual_analysis_results)} chars")
        
        return state
    
    def _strategist_node(self, state: AgentState) -> AgentState:
        """
        Strategist node that synthesizes insights and formulates recommendations.
        
        Args:
            state: Current state.
            
        Returns:
            Updated state.
        """
        query = state["query"]
        research_results = state.get("research_results", {}).get("content", "")
        analysis_results = state.get("analysis_results", {}).get("content", "")
        visual_analysis_results = state.get("visual_analysis_results", {}).get("content", "")
        
        # Build inputs dictionary with available results
        inputs = {}
        if research_results:
            inputs["Research"] = research_results
        if analysis_results:
            inputs["Analysis"] = analysis_results
        if visual_analysis_results:
            inputs["Visual Analysis"] = visual_analysis_results
        
        # Run the strategist agent
        strategy_results = self.strategist_agent.synthesize(inputs)
        
        # Update the state
        state["strategy_results"] = {
            "content": strategy_results,
            "timestamp": str(uuid.uuid4()),
        }
        state["current_agent"] = "strategist"
        state["messages"].append(AIMessage(content=strategy_results))
        
        if self.verbose:
            logger.info(f"Strategist completed synthesis: {len(strategy_results)} chars")
        
        return state
    
    def _critic_node(self, state: AgentState) -> AgentState:
        """
        Critic node that evaluates and improves outputs.
        
        Args:
            state: Current state.
            
        Returns:
            Updated state.
        """
        query = state["query"]
        strategy_results = state.get("strategy_results", {}).get("content", "")
        research_results = state.get("research_results", {}).get("content", "")
        
        # Run the critic agent
        critique_prompt = f"""I need you to evaluate the following strategic insights and recommendations:

Strategic Insights:
{strategy_results}

Original Query: {query}

Research Context:
{research_results}

Please provide a thorough critique that includes:
1. Assessment of factual accuracy and consistency
2. Identification of gaps, inconsistencies, or weaknesses
3. Evaluation of the quality and completeness of the recommendations
4. Specific suggestions for improvement
5. Overall assessment of whether the output effectively addresses the original query

Be constructively critical and focus on specific improvements that would enhance the quality of the output."""
        
        critique_results = self.critic_agent.evaluate(strategy_results, research_results)
        
        # Update the state
        state["critique_results"] = {
            "content": critique_results,
            "timestamp": str(uuid.uuid4()),
        }
        state["current_agent"] = "critic"
        state["messages"].append(AIMessage(content=critique_results))
        
        if self.verbose:
            logger.info(f"Critic completed evaluation: {len(critique_results)} chars")
        
        return state
    
    
    def _finalizer_node(self, state: AgentState) -> AgentState:
        """
        Finalizer node that creates the final response.
        
        Args:
            state: Current state.
            
        Returns:
            Updated state with final response.
        """
        query = state["query"]
        strategy_results = state.get("strategy_results", {}).get("content", "")
        critique_results = state.get("critique_results", {}).get("content", "")
        
        # Create finalizer prompt
        finalizer_prompt = f"""Based on the original query, the strategic insights, and the critique, please create a final comprehensive response:

Original Query: 
{query}

Strategic Insights:
{strategy_results}

Critique and Suggestions for Improvement:
{critique_results}

Please synthesize this information into a comprehensive final response that:
1. Directly addresses the original query
2. Incorporates the key insights from the strategic analysis
3. Addresses the improvements suggested in the critique
4. Provides a well-structured, clear, and actionable response
5. Includes appropriate citations or references to BCG's sustainability reports

Focus on creating a polished, professional response that effectively answers the query while incorporating the feedback from the critique."""
        
        # Use the LLM to create the final response
        final_response = self.llm.invoke(finalizer_prompt)
        
        # Update the state
        state["final_response"] = final_response.content
        state["current_agent"] = "finalizer"
        state["messages"].append(AIMessage(content=final_response.content))
        
        if self.verbose:
            logger.info(f"Finalizer created final response: {len(final_response.content)} chars")
        
        return state
    
    def _route_after_coordinator(self, state: AgentState) -> str:
        """
        Determine the next agent after the coordinator.
        
        Args:
            state: Current state.
            
        Returns:
            Name of the next agent.
        """
        return state["next_agent"]
    
    def _route_after_researcher(self, state: AgentState) -> str:
        """
        Determine the next agent after the researcher.
        
        Args:
            state: Current state.
            
        Returns:
            Name of the next agent.
        """
        query = state["query"].lower()
        
        # Check if query specifically asks for visual analysis
        if any(term in query for term in ["chart", "graph", "visual", "image", "picture", "diagram"]):
            return "visual_analyst"
        
        # Check if query asks for data analysis
        elif any(term in query for term in ["analyze", "analysis", "trend", "compare", "data", "metric", "statistics"]):
            return "analyst"
        
        # Otherwise, go straight to strategist
        else:
            return "strategist"
    
    def _route_after_analyst(self, state: AgentState) -> str:
        """
        Determine the next agent after the analyst.
        
        Args:
            state: Current state.
            
        Returns:
            Name of the next agent.
        """
        query = state["query"].lower()
        
        # Check if we also need visual analysis
        if any(term in query for term in ["chart", "graph", "visual", "image", "picture", "diagram"]):
            return "visual_analyst"
        else:
            return "strategist"
    
    def _route_after_critic(self, state: AgentState) -> str:
        """
        Determine the next agent after the critic.
        
        Args:
            state: Current state.
            
        Returns:
            Name of the next agent or 'finalizer' to finish.
        """
        critique = state.get("critique_results", {}).get("content", "").lower()
        
        # Check if critique suggests major revision needed
        needs_revision = False
        
        # Check for phrases indicating significant issues
        revision_indicators = [
            "major issues", 
            "significant problems", 
            "requires substantial revision",
            "missing critical information",
            "factually incorrect",
            "completely inadequate",
            "needs to be redone"
        ]
        
        for indicator in revision_indicators:
            if indicator in critique:
                needs_revision = True
                break
        
        if needs_revision:
            # Determine which agent needs to revise
            if "more research" in critique or "missing information" in critique:
                return "researcher"
            elif "analysis is flawed" in critique or "data interpretation" in critique:
                return "analyst"
            elif "visual interpretation" in critique or "misreading charts" in critique:
                return "visual_analyst"
            else:
                # Default to strategist for revisions
                return "strategist"
        else:
            # No major revision needed, proceed to finalizer
            return "finalizer"
    
    def process_query(self, query: str) -> str:
        """
        Process a query through the multi-agent system.
        
        Args:
            query: User query.
            
        Returns:
            Final response from the multi-agent system.
        """
        logger.info(f"Processing query: '{query}'")
        
        # Initialize the state
        initial_state = AgentState(
            messages=[],
            query=query,
            context={},
            research_results=None,
            analysis_results=None,
            visual_analysis_results=None,
            strategy_results=None,
            critique_results=None,
            final_response=None,
            current_agent="",
            next_agent=None,
        )
        
        # Execute the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            # Extract the final response
            final_response = final_state["final_response"]
            
            if not final_response:
                logger.error("No final response generated")
                return "Error: No response was generated for your query."
            
            logger.info("Query processing completed successfully")
            return final_response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error: An error occurred while processing your query. Error details: {str(e)}"


    