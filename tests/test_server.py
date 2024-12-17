import pytest
import vcr
from fastapi.testclient import TestClient
from server import app, WORKING_DIR
import os
import shutil

# Configure VCR
vcr = vcr.VCR(
    cassette_library_dir="tests/fixtures/vcr_cassettes",
    record_mode="once",
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
    filter_headers=["authorization"],  # Don't record API keys
)

@pytest.fixture(autouse=True)
async def cleanup():
    """Clean up the test directory before and after each test"""
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
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@vcr.use_cassette("tests/fixtures/vcr_cassettes/test_add_document.yaml")
def test_add_document():
    document = {
        "content": "This is a test document.",
        "metadata": {"source": "test"}
    }
    response = client.post("/documents", json=document)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

@vcr.use_cassette("tests/fixtures/vcr_cassettes/test_add_documents_batch.yaml")
def test_add_documents_batch():
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
    # Test file upload
    test_content = b"This is a test file content"
    files = {"file": ("test.txt", test_content, "text/plain")}
    data = {"file_type": "text"}

    response = client.post("/documents/file", files=files, data=data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

