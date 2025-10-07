import os
import hashlib
import logging
from PyPDF2 import PdfReader
from llama_index.core.schema import Document
from llama_index.core.text_splitter import SentenceSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_pdf_id(pdf_path: str) -> str:
    """
    Generates a unique ID for a PDF file based on its content hash.
    This ensures that if the PDF content changes, it's treated as a new file.
    """
    try:
        hasher = hashlib.md5()
        with open(pdf_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except IOError as e:
        logging.error(f"Could not read file to generate ID: {e}")
        return None

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a given PDF file.
    Includes basic validation to check for corrupted or empty PDFs.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"File not found at '{pdf_path}'")
        return None

    try:
        logging.info(f"Attempting to parse: {pdf_path}")
        reader = PdfReader(pdf_path)

        # --- Validation ---
        if len(reader.pages) == 0:
            logging.warning(f"PDF file '{os.path.basename(pdf_path)}' contains no pages. It may be empty or corrupted.")
            return None

        # Check if the document is encrypted and cannot be opened
        if reader.is_encrypted:
            logging.warning(f"PDF file '{os.path.basename(pdf_path)}' is encrypted and cannot be processed.")
            return None # Or attempt decryption if a password is provided

        text = ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as page_error:
                logging.warning(f"Could not extract text from page {i+1} in '{pdf_path}'. It might be a scanned image. Error: {page_error}")
                continue # Move to the next page

        if not text.strip():
            logging.warning(f"No text content could be extracted from '{os.path.basename(pdf_path)}'. The document might be image-based or empty.")
            return None

        logging.info(f"Successfully extracted text from {os.path.basename(pdf_path)}")
        return text
    except Exception as e:
        logging.error(f"Failed to read or process PDF '{pdf_path}': {e}. The file may be corrupted.")
        return None

def chunk_text(text: str):
    """
    Splits a long text into smaller, manageable chunks (nodes).
    """
    if not text:
        logging.warning("Cannot chunk empty text.")
        return []

    try:
        document = Document(text=text)
        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents([document])

        if not nodes:
            logging.warning("Text splitting resulted in zero nodes. The text may be too short for the current chunk settings.")
        else:
            logging.info(f"Split text into {len(nodes)} nodes (chunks).")

        return nodes
    except Exception as e:
        logging.error(f"An unexpected error occurred during text chunking: {e}")
        return []