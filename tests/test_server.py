"""Test suite for LightRAG server API endpoints.

This module provides integration tests for the FastAPI server endpoints,
testing various document operations, querying capabilities, and file handling.
It uses VCR.py to record and replay HTTP interactions, ensuring consistent
test behavior across runs.

Key Features Tested:
- Health check endpoint
- Document addition (single and batch)
- Document querying
- File upload handling
- Error scenarios

Test Configuration:
- Uses TestClient for FastAPI testing
- VCR.py for HTTP interaction recording
- Automatic test directory cleanup
- Request/response validation
"""

import pytest
import vcr
from fastapi.testclient import TestClient
from server import app, WORKING_DIR
import os
import shutil

# Configure VCR for HTTP interaction recording
vcr = vcr.VCR(
    cassette_library_dir="tests/fixtures/vcr_cassettes",
    record_mode="once",
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
    filter_headers=["authorization"],  # Don't record API keys
)

@pytest.fixture(autouse=True)
async def cleanup():
    """Fixture to manage test directory cleanup.

    Ensures a clean test environment by:
    - Removing existing test directory before each test
    - Creating a fresh test directory
    - Cleaning up after test completion

    This fixture runs automatically for all tests in this module.
    """
    # Setup
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR)

    yield

    # Cleanup
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint.

    Verifies that:
    - Endpoint returns 200 status code
    - Response indicates healthy status
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@vcr.use_cassette("tests/fixtures/vcr_cassettes/test_add_document.yaml")
def test_add_document():
    """Test single document addition endpoint.

    Verifies:
    - Document can be added successfully
    - Response format is correct
    - Status code indicates success
    """
    document = {
        "content": "This is a test document.",
        "metadata": {"source": "test"}
    }
    response = client.post("/documents", json=document)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

@vcr.use_cassette("tests/fixtures/vcr_cassettes/test_add_documents_batch.yaml")
def test_add_documents_batch():
    """Test batch document addition endpoint.

    Verifies:
    - Multiple documents can be added in one request
    - Response format is correct
    - Status code indicates success
    """
    documents = [
        {
            "content": "First test document.",
            "metadata": {"source": "test1"}
        },
        {
            "content": "Second test document.",
            "metadata": {"source": "test2"}
        }
    ]
    response = client.post("/documents/batch", json=documents)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

@vcr.use_cassette("tests/fixtures/vcr_cassettes/test_query.yaml")
def test_query():
    """Test document querying endpoint.

    Verifies:
    - Query execution against added documents
    - Hybrid search mode functionality
    - Response format and success status
    - Context retrieval options
    """
    # First add a document to query against
    document = {
        "content": "The capital of France is Paris.",
        "metadata": {"source": "test"}
    }
    client.post("/documents", json=document)

    query_request = {
        "query": "What is the capital of France?",
        "mode": "hybrid",
        "only_need_context": False
    }
    response = client.post("/query", json=query_request)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

@vcr.use_cassette(
    "tests/fixtures/vcr_cassettes/test_upload_file_new.yaml",
    filter_headers=["authorization"],
    record_mode="once",
    match_on=["method", "scheme", "host", "port", "path", "query"]
)
def test_upload_file():
    """Test file upload endpoint.

    Verifies:
    - File upload functionality
    - Content type handling
    - Response format and success status
    - File type parameter processing
    """
    test_content = b"This is a test file content"
    files = {"file": ("test.txt", test_content, "text/plain")}
    data = {"file_type": "text"}

    response = client.post("/documents/file", files=files, data=data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

