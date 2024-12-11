import os
import tempfile
import pytest
import vcr

from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete

# Create a fixture for test data
@pytest.fixture
def sample_text():
    return """
    It was the best of times, it was the worst of times, it was the age of wisdom,
    it was the age of foolishness, it was the epoch of belief, it was the epoch of
    incredulity, it was the season of Light, it was the season of Darkness.
    """

@pytest.fixture
def multiple_documents():
    return [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
    ]

# Use VCR to record OpenAI responses
@vcr.use_cassette(
    'tests/fixtures/vcr_cassettes/test_lightrag_basic.yaml',
    filter_headers=['authorization'],
    record_mode='once'
)
def test_lightrag_basic_operations(sample_text):
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
    """Test handling multiple documents and retrieving relevant information."""
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
    """Test that repeated queries with same parameters return consistent results."""
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
