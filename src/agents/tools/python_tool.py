"""
Python code execution tool for BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
import sys
import io
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import traceback

from langchain_core.tools import BaseTool, ToolException

logger = logging.getLogger(__name__)

class PythonInput(BaseModel):
    """Input schema for the Python tool."""
    code: str = Field(..., description="The Python code to execute")
    use_allowed_imports: bool = Field(True, description="Whether to restrict imports to the allowed list")

class PythonTool(BaseTool):
    """
    Tool for executing Python code for data analysis.
    
    This tool allows agents to perform data analysis, create visualizations,
    and generate insights by executing Python code.
    """
    
    name = "python_executor"
    description = "Execute Python code for data analysis, visualization, or other computations"
    args_schema = PythonInput
    
    def __init__(
        self,
        allowed_imports: Optional[List[str]] = None,
        globals_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Python tool.
        
        Args:
            allowed_imports: List of allowed import modules.
            globals_dict: Dictionary of global variables to make available to the code.
        """
        super().__init__()
        
        # Default allowed imports for data analysis
        self.allowed_imports = allowed_imports or [
            "numpy", "pandas", "matplotlib", "seaborn", 
            "math", "statistics", "datetime", "re", 
            "collections", "json", "csv", "io"
        ]
        
        # Set up globals dictionary
        self.globals_dict = globals_dict or {}
        
        # Add commonly used libraries to globals
        self._setup_default_globals()
    
    def _setup_default_globals(self) -> None:
        """Set up default globals for Python execution."""
        try:
            # Add pandas and numpy by default
            import pandas as pd
            import numpy as np
            import matplotlib
            import matplotlib.pyplot as plt
            import seaborn as sns
            import json
            
            # Configure matplotlib to work in various environments
            matplotlib.use('Agg')
            
            # Add to globals
            self.globals_dict.update({
                "pd": pd,
                "np": np,
                "plt": plt,
                "sns": sns,
                "json": json,
            })
            
            logger.info("Default globals set up for Python executor")
        except ImportError as e:
            logger.warning(f"Could not import some libraries for Python executor: {str(e)}")
    
    def _run(self, code: str, use_allowed_imports: bool = True) -> str:
        """
        Execute Python code.
        
        Args:
            code: Python code to execute.
            use_allowed_imports: Whether to restrict imports to the allowed list.
            
        Returns:
            Output of the code execution.
        """
        logger.info("Executing Python code")
        
        # Check for potentially harmful operations
        forbidden_terms = [
            "os.system", "subprocess", "eval(", "exec(", 
            "__import__", "importlib", "open(", "file(",
            "globals()", "locals()",
            "delete", "remove", "rmdir", "unlink",
            "sys.modules", "sys.path"
        ]
        
        for term in forbidden_terms:
            if term in code:
                error_message = f"Code contains potentially unsafe operation: {term}"
                logger.error(error_message)
                raise ToolException(error_message)
        
        # Check imports if restriction is enabled
        if use_allowed_imports:
            import_lines = [line.strip() for line in code.split('\n') 
                           if line.strip().startswith(('import ', 'from ')) 
                           and not line.strip().startswith('#')]
            
            for import_line in import_lines:
                import_module = None
                
                if import_line.startswith('import '):
                    # Handle "import numpy as np" or "import numpy"
                    parts = import_line.split(' as ')[0].split(' ', 1)[1].split(',')
                    import_module = parts[0].strip()
                elif import_line.startswith('from '):
                    # Handle "from numpy import array" or "from numpy.random import *"
                    import_module = import_line.split(' import ')[0].split(' ', 1)[1].strip()
                
                if import_module and not any(import_module.startswith(allowed) for allowed in self.allowed_imports):
                    error_message = f"Import not allowed: {import_module}. Allowed imports: {', '.join(self.allowed_imports)}"
                    logger.error(error_message)
                    raise ToolException(error_message)
        
        # Execute the code
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = io.StringIO()
        redirected_error = io.StringIO()
        sys.stdout = redirected_output
        sys.stderr = redirected_error
        
        try:
            # Execute the code with the prepared globals
            exec(code, self.globals_dict)
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Get the output and error
            output_str = redirected_output.getvalue()
            error_str = redirected_error.getvalue()
            
            # Return the combined output
            result = ""
            if output_str:
                result += f"Output:\n{output_str}\n"
            if error_str:
                result += f"Errors/Warnings:\n{error_str}\n"
            
            # Check if there's a plot to display
            if 'plt' in self.globals_dict and 'plt.show()' in code:
                result += "\n[Note: A plot was generated but cannot be displayed directly in text. The plot data is available in the Python context.]"
            
            if not result:
                result = "Code executed successfully with no output."
            
            logger.info("Python code execution completed successfully")
            return result
        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            error_message = f"Error executing Python code: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_message)
            
            # Include any output that was generated before the error
            output_str = redirected_output.getvalue()
            if output_str:
                error_message = f"Partial output before error:\n{output_str}\n\n{error_message}"
            
            raise ToolException(error_message)