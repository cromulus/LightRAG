import pytest
import asyncio
import os
from lightrag import LightRAG
from lightrag.kg.postgres_impl import PostgresKVStorage
from lightrag.storage import (
    NanoVectorDBStorage,
    NetworkXStorage,
)

def dummy_embedding_func(text):
    """Dummy embedding function that returns a fixed vector"""
    return [0.1] * 10

@pytest.fixture
async def pg_config():
    return {
        "postgres": {
            "host": "localhost",
            "port": 5432,
            "user": "lightrag_test",
            "password": "lightrag_test",
            "database": "lightrag_test"
        }
    }

@pytest.fixture
async def lightrag_instance(pg_config):
    # Create a temporary working directory
    working_dir = "test_lightrag_postgres"
    os.makedirs(working_dir, exist_ok=True)

    # Initialize LightRAG with PostgreSQL storage
    rag = LightRAG(
        working_dir=working_dir,
        kv_storage="PostgresKVStorage",
        vector_storage="PostgresKVStorage",  # Using same storage for vectors
        graph_storage="PostgresKVStorage",   # Using same storage for graph
        embedding_func=dummy_embedding_func,
        addon_params={"postgres": pg_config["postgres"]}  # Pass PostgreSQL config through addon_params
    )

    yield rag

    # Cleanup
    await rag.close()
    if os.path.exists(working_dir):
        import shutil
        shutil.rmtree(working_dir)

@pytest.mark.asyncio
async def test_postgres_e2e_basic_workflow(lightrag_instance):
    """Test basic end-to-end workflow with PostgreSQL storage"""
    # Test data
    documents = [
        {
            "id": "doc1",
            "content": "The quick brown fox jumps over the lazy dog",
            "metadata": {"type": "test", "category": "animals"}
        },
        {
            "id": "doc2",
            "content": "Python is a versatile programming language",
            "metadata": {"type": "test", "category": "programming"}
        }
    ]

    # Add documents
    await lightrag_instance.add_documents(documents)

    # Test retrieval by ID
    doc = await lightrag_instance.get_document("doc1")
    assert doc["content"] == "The quick brown fox jumps over the lazy dog"
    assert doc["metadata"]["category"] == "animals"

    # Test batch retrieval
    docs = await lightrag_instance.get_documents(["doc1", "doc2"])
    assert len(docs) == 2
    assert docs[0]["metadata"]["type"] == "test"
    assert docs[1]["content"] == "Python is a versatile programming language"

    # Test filtering
    filtered_docs = await lightrag_instance.get_documents(
        ["doc1", "doc2"],
        fields={"content"}
    )
    assert len(filtered_docs) == 2
    assert "content" in filtered_docs[0]
    assert "metadata" not in filtered_docs[0]

@pytest.mark.asyncio
async def test_postgres_e2e_document_updates(lightrag_instance):
    """Test document update operations with PostgreSQL storage"""
    # Initial document
    doc = {
        "id": "update_test",
        "content": "Original content",
        "metadata": {"version": 1}
    }
    await lightrag_instance.add_document(doc)

    # Update document
    updated_doc = {
        "id": "update_test",
        "content": "Updated content",
        "metadata": {"version": 2}
    }
    await lightrag_instance.add_document(updated_doc)

    # Verify update
    result = await lightrag_instance.get_document("update_test")
    assert result["content"] == "Updated content"
    assert result["metadata"]["version"] == 2

@pytest.mark.asyncio
async def test_postgres_e2e_nonexistent_documents(lightrag_instance):
    """Test handling of nonexistent documents with PostgreSQL storage"""
    # Try to get nonexistent document
    doc = await lightrag_instance.get_document("nonexistent")
    assert doc is None

    # Try to get mix of existing and nonexistent documents
    await lightrag_instance.add_document({
        "id": "existing",
        "content": "This document exists",
        "metadata": {}
    })

    docs = await lightrag_instance.get_documents(["existing", "nonexistent"])
    assert len(docs) == 2
    assert docs[0] is not None
    assert docs[1] is None

@pytest.mark.asyncio
async def test_postgres_e2e_complex_operations(lightrag_instance):
    """Test complex operations with PostgreSQL storage including batch operations and metadata filtering"""
    # Test batch document addition with varied metadata
    documents = [
        {
            "id": f"batch_{i}",
            "content": f"Content for document {i}",
            "metadata": {
                "type": "batch",
                "priority": i % 3,
                "tags": ["tag1", "tag2"] if i % 2 == 0 else ["tag3"],
                "timestamp": f"2024-01-{i+1:02d}"
            }
        } for i in range(5)
    ]

    # Add documents in batch
    await lightrag_instance.add_documents(documents)

    # Test partial field retrieval with specific metadata
    filtered_docs = await lightrag_instance.get_documents(
        ["batch_0", "batch_1", "batch_2"],
        fields={"content", "metadata.priority", "metadata.tags"}
    )
    assert len(filtered_docs) == 3
    assert all("content" in doc for doc in filtered_docs if doc)
    assert all("metadata" in doc for doc in filtered_docs if doc)
    assert all("timestamp" not in doc["metadata"] for doc in filtered_docs if doc)

    # Test concurrent updates
    update_docs = [
        {
            "id": "batch_0",
            "content": "Updated content 0",
            "metadata": {"priority": 9, "updated": True}
        },
        {
            "id": "batch_1",
            "content": "Updated content 1",
            "metadata": {"priority": 8, "updated": True}
        }
    ]

    # Perform concurrent updates
    await asyncio.gather(
        lightrag_instance.add_document(update_docs[0]),
        lightrag_instance.add_document(update_docs[1])
    )

    # Verify updates
    updated_docs = await lightrag_instance.get_documents(["batch_0", "batch_1"])
    assert updated_docs[0]["content"] == "Updated content 0"
    assert updated_docs[0]["metadata"]["priority"] == 9
    assert updated_docs[0]["metadata"]["updated"] is True
    assert updated_docs[1]["content"] == "Updated content 1"
    assert updated_docs[1]["metadata"]["priority"] == 8
    assert updated_docs[1]["metadata"]["updated"] is True

    # Test mixed existing/non-existing document retrieval
    mixed_docs = await lightrag_instance.get_documents(["batch_0", "nonexistent", "batch_1"])
    assert len(mixed_docs) == 3
    assert mixed_docs[0] is not None
    assert mixed_docs[1] is None
    assert mixed_docs[2] is not None

    # Test empty batch operations
    empty_results = await lightrag_instance.get_documents([])
    assert empty_results == []

    # Test large document update
    large_content = "x" * 1000000  # 1MB of content
    large_doc = {
        "id": "large_doc",
        "content": large_content,
        "metadata": {"size": "large"}
    }
    await lightrag_instance.add_document(large_doc)
    retrieved_large_doc = await lightrag_instance.get_document("large_doc")
    assert len(retrieved_large_doc["content"]) == 1000000
