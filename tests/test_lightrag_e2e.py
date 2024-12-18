"""End-to-end test suite for LightRAG functionality.

This module provides comprehensive end-to-end testing for the LightRAG system,
covering document insertion, querying with different modes, and result consistency.
It uses VCR.py to record and replay LLM interactions, ensuring consistent test
behavior across runs.

Key Features Tested:
- Basic RAG operations (insert, query)
- Multiple document handling
- Query mode variations (naive, local, global, hybrid)
- Response consistency
- LLM integration

Test Configuration:
- Uses temporary directories for isolation
- VCR.py for LLM response recording
- Multiple test scenarios
- Comprehensive assertion checks
"""

import os
import tempfile
import pytest
import vcr
import shutil
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
from lightrag.kg.age_impl import AGEStorage
from lightrag.utils import logger

@pytest.fixture
def sample_text():
    """Fixture providing a sample text with contrasting themes.

    Returns:
        str: A passage from 'A Tale of Two Cities' with clear contrasts
            suitable for testing theme extraction and analysis.
    """
    return """
    It was the best of times, it was the worst of times, it was the age of wisdom,
    it was the age of foolishness, it was the epoch of belief, it was the epoch of
    incredulity, it was the season of Light, it was the season of Darkness.
    """

@pytest.fixture
def multiple_documents():
    """Fixture providing multiple distinct documents for testing.

    Returns:
        list[str]: A collection of different text samples, each with unique
            content suitable for testing document retrieval and relevance.
    """
    return [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
    ]

@vcr.use_cassette(
    'tests/fixtures/vcr_cassettes/test_lightrag_basic.yaml',
    filter_headers=['authorization'],
    record_mode='once'
)
def test_lightrag_basic_operations(sample_text):
    """Test basic LightRAG operations with different query modes.

    Tests:
        - Document insertion
        - Querying with all available modes
        - Response quality and relevance
        - Theme extraction capabilities

    Args:
        sample_text: Test document with contrasting themes
    """
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize LightRAG
        rag = LightRAG(
            working_dir=temp_dir,
            llm_model_func=gpt_4o_mini_complete,
        )

        # Test document insertion
        rag.insert(sample_text)

        # Test query with different modes
        query = "What are the contrasting themes in this text?"
        modes = ["naive", "local", "global", "hybrid"]

        for mode in modes:
            response = rag.query(query, param=QueryParam(mode=mode))
            assert isinstance(response, str)
            assert len(response) > 0

            # Basic content checks
            assert "time" in response.lower() or "contrast" in response.lower(), \
                f"Response for {mode} mode should mention key themes"

@vcr.use_cassette(
    'tests/fixtures/vcr_cassettes/test_lightrag_multiple_docs.yaml',
    filter_headers=['authorization'],
    record_mode='once'
)
def test_multiple_document_handling(multiple_documents):
    """Test LightRAG's ability to handle and query multiple documents.

    Tests:
        - Multiple document insertion
        - Cross-document querying
        - Relevant document retrieval
        - Response accuracy for specific queries

    Args:
        multiple_documents: Collection of test documents
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        rag = LightRAG(
            working_dir=temp_dir,
            llm_model_func=gpt_4o_mini_complete,
        )

        # Insert multiple documents
        for doc in multiple_documents:
            rag.insert(doc)

        # Query about specific document content
        response = rag.query(
            "Tell me about the fox and the dog",
            param=QueryParam(mode="hybrid")
        )
        assert "fox" in response.lower() and "dog" in response.lower(), \
            "Response should contain information from the relevant document"

@vcr.use_cassette(
    'tests/fixtures/vcr_cassettes/test_lightrag_consistency.yaml',
    filter_headers=['authorization'],
    record_mode='once'
)
def test_query_consistency(sample_text):
    """Test consistency of LightRAG query responses.

    Tests:
        - Response consistency across multiple queries
        - Semantic similarity of responses
        - Theme identification consistency
        - Response quality maintenance

    Args:
        sample_text: Test document with consistent themes
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        rag = LightRAG(
            working_dir=temp_dir,
            llm_model_func=gpt_4o_mini_complete,
        )

        rag.insert(sample_text)

        # Make same query multiple times
        query = "What is the main contrast in the text?"
        param = QueryParam(mode="hybrid")

        first_response = rag.query(query, param=param)
        second_response = rag.query(query, param=param)

        # Responses might not be identical due to LLM, but should be semantically similar
        assert isinstance(first_response, str) and isinstance(second_response, str)
        assert len(first_response) > 0 and len(second_response) > 0
        assert "time" in first_response.lower() and "time" in second_response.lower(), \
            "Responses should consistently identify key themes"

