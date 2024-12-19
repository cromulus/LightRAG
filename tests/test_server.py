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
    """Test single document addition endpoint without user_id.

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

@vcr.use_cassette("tests/fixtures/vcr_cassettes/test_add_document_with_user.yaml")
def test_add_document_with_user():
    """Test single document addition endpoint with user_id.

    Verifies:
    - Document can be added successfully with user context
    - Response format is correct
    - Status code indicates success
    """
    document = {
        "content": "This is a test document for user1.",
        "metadata": {"source": "test"}
    }
    headers = {"X-User-ID": "user1"}
    response = client.post("/documents", json=document, headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

@vcr.use_cassette("tests/fixtures/vcr_cassettes/test_add_documents_batch.yaml")
def test_add_documents_batch():
    """Test batch document addition endpoint without user_id.

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

@vcr.use_cassette("tests/fixtures/vcr_cassettes/test_add_documents_batch_with_user.yaml")
def test_add_documents_batch_with_user():
    """Test batch document addition endpoint with user_id.

    Verifies:
    - Multiple documents can be added in one request with user context
    - Response format is correct
    - Status code indicates success
    """
    documents = [
        {
            "content": "First test document for user1.",
            "metadata": {"source": "test1"}
        },
        {
            "content": "Second test document for user1.",
            "metadata": {"source": "test2"}
        }
    ]
    headers = {"X-User-ID": "user1"}
    response = client.post("/documents/batch", json=documents, headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

@vcr.use_cassette("tests/fixtures/vcr_cassettes/test_query_documents.yaml")
def test_query_documents():
    """Test document querying without user_id."""
    # First add a test document
    document = {
        "content": "Test document about machine learning and AI",
        "metadata": {"source": "test"}
    }
    client.post("/documents", json=document)

    # Then query
    query = {"query": "machine learning", "top_k": 5}
    response = client.post("/query", json=query)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "success"
    assert "data" in data
    assert isinstance(data["data"], str)
    assert "machine learning" in data["data"].lower()

@vcr.use_cassette("tests/fixtures/vcr_cassettes/test_query_documents_with_user.yaml")
def test_query_documents_with_user():
    """Test document querying with user_id."""
    # First add a test document
    document = {
        "content": "Test document about machine learning and AI",
        "metadata": {"source": "test"}
    }
    headers = {"X-User-ID": "user1"}
    client.post("/documents", json=document, headers=headers)

    # Then query
    query = {"query": "machine learning", "top_k": 5}
    response = client.post("/query", json=query, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "success"
    assert "data" in data
    assert isinstance(data["data"], str)
    assert "machine learning" in data["data"].lower()

@vcr.use_cassette("tests/fixtures/vcr_cassettes/test_query_documents_user_isolation.yaml")
def test_query_documents_user_isolation():
    """Test that documents are properly isolated between users."""
    # Add document for user1
    doc1 = {
        "content": "Unique document for user1 about pandas.",
        "metadata": {"source": "test"}
    }
    headers1 = {"X-User-ID": "user1"}
    response = client.post("/documents", json=doc1, headers=headers1)
    assert response.status_code == 200

    # Add document for user2
    doc2 = {
        "content": "Unique document for user2 about koalas.",
        "metadata": {"source": "test"}
    }
    headers2 = {"X-User-ID": "user2"}
    response = client.post("/documents", json=doc2, headers=headers2)
    assert response.status_code == 200

    # Query for pandas with user1
    query = {"query": "Tell me about pandas", "top_k": 5}
    response1 = client.post("/query", json=query, headers=headers1)
    data1 = response1.json()

    # Query for koalas with user2
    query2 = {"query": "Tell me about koalas", "top_k": 5}
    response2 = client.post("/query", json=query2, headers=headers2)
    data2 = response2.json()

    # Check that each user only sees their own content
    assert "pandas" in data1["data"].lower()
    assert "koalas" in data2["data"].lower()
    assert "koalas" not in data1["data"].lower()
    assert "pandas" not in data2["data"].lower()

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

