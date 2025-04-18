"""
Main entry point for the BCG Multi-Agent & Multimodal AI Platform.
"""
import logging
from configs.config import LOG_LEVEL, DEBUG

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting BCG Multi-Agent & Multimodal AI Platform")
    logger.debug(f"Debug mode is {'enabled' if DEBUG else 'disabled'}")
    
    # This will be expanded as we build out the application
    print("BCG Multi-Agent & Multimodal AI Platform initialized")