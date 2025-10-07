import os
import pytest
from src.pdf_processor import get_pdf_id, extract_text_from_pdf, chunk_text

# --- Fixtures ---

@pytest.fixture(scope="module")
def test_pdf_path():
    return os.path.join(os.path.dirname(__file__), "test_document.pdf")

@pytest.fixture(scope="module")
def empty_pdf_path():
    return os.path.join(os.path.dirname(__file__), "empty_document.pdf")

@pytest.fixture(scope="module")
def image_pdf_path():
    return os.path.join(os.path.dirname(__file__), "image_only_document.pdf")

# --- Unit Tests for pdf_processor.py ---

def test_get_pdf_id_consistency(test_pdf_path):
    """
    Tests that get_pdf_id returns the same ID for the same file.
    """
    pdf_id_1 = get_pdf_id(test_pdf_path)
    pdf_id_2 = get_pdf_id(test_pdf_path)
    assert pdf_id_1 is not None
    assert pdf_id_1 == pdf_id_2

def test_get_pdf_id_uniqueness(test_pdf_path, empty_pdf_path):
    """
    Tests that get_pdf_id returns different IDs for different files.
    """
    pdf_id_1 = get_pdf_id(test_pdf_path)
    pdf_id_2 = get_pdf_id(empty_pdf_path)
    assert pdf_id_1 is not None
    assert pdf_id_2 is not None
    assert pdf_id_1 != pdf_id_2

def test_extract_text_from_valid_pdf(test_pdf_path):
    """
    Tests text extraction from a standard, valid PDF.
    """
    text = extract_text_from_pdf(test_pdf_path)
    assert text is not None
    assert "Artificial Intelligence (AI)" in text
    assert "ethics of creating artificial beings" in text
    assert len(text.strip()) > 500

def test_extract_text_from_empty_pdf(empty_pdf_path):
    """
    Tests that an empty PDF returns None, with appropriate warnings.
    """
    text = extract_text_from_pdf(empty_pdf_path)
    assert text is None

def test_extract_text_from_image_pdf(image_pdf_path):
    """
    Tests that an image-only PDF (no extractable text) returns None.
    """
    text = extract_text_from_pdf(image_pdf_path)
    assert text is None

def test_extract_text_from_non_existent_file():
    """
    Tests that a non-existent file path is handled gracefully.
    """
    text = extract_text_from_pdf("non_existent_file.pdf")
    assert text is None

def test_chunk_text_valid():
    """
    Tests that text is split into a list of Node objects.
    """
    sample_text = "This is a test sentence. " * 100
    nodes = chunk_text(sample_text)
    assert nodes is not None
    assert len(nodes) > 0
    # The exact number depends on chunk_size, but it should be more than 1 for this text
    assert len(nodes) > 1
    assert "This is a test sentence." in nodes[0].get_content()

def test_chunk_text_empty_input():
    """
    Tests that chunk_text handles empty or None input correctly.
    """
    nodes_empty = chunk_text("")
    assert nodes_empty == []

    nodes_none = chunk_text(None)
    assert nodes_none == []

def test_chunk_text_short_input():
    """
    Tests that text shorter than the chunk size results in a single node.
    """
    sample_text = "This is a single sentence that is shorter than the chunk size."
    nodes = chunk_text(sample_text)
    assert len(nodes) == 1
    assert nodes[0].get_content() == sample_text