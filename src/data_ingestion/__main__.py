"""
Command-line script for processing BCG Sustainability Reports.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from configs.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.data_ingestion.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process BCG Sustainability Reports")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Input directory containing PDF files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="Glob pattern to match PDF files",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Size of text chunks for splitting",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="Overlap between consecutive chunks",
    )
    parser.add_argument(
        "--extract-visuals",
        action="store_true",
        help="Extract visual elements from PDFs",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence score for visual elements",
    )
    parser.add_argument(
        "--no-unstructured",
        action="store_true",
        help="Disable using unstructured library for text extraction",
    )
    return parser.parse_args()

def main():
    """Main function for processing BCG Sustainability Reports."""
    args = parse_args()
    
    # Create the document processor
    document_processor = DocumentProcessor(
        use_unstructured=not args.no_unstructured,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        extract_visuals=args.extract_visuals,
        confidence_threshold=args.confidence_threshold,
    )
    
    # Process all PDF files in the input directory
    processed_docs = document_processor.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_pattern=args.pattern,
    )
    
    logger.info(f"Successfully processed {len(processed_docs)} documents")
    for doc in processed_docs:
        logger.info(f" - {doc.filename}: {len(doc.text_chunks)} chunks, {len(doc.visual_elements)} visual elements")

if __name__ == "__main__":
    main()