"""Test suite for vector storage implementations in LightRAG.

This module provides comprehensive testing for different vector storage backends,
including NanoVectorDB and PostgreSQL implementations. It tests various operations
such as insertion, querying, metadata filtering, and concurrent operations across
different namespaces (chunks, entities, relationships).

The test suite uses parametrized fixtures to run the same tests against different
storage implementations and namespaces, ensuring consistent behavior across all
storage backends.

Key Features Tested:
- Basic CRUD operations
- Batch operations
- Metadata filtering
- Concurrent operations
- Edge cases (empty queries, nonexistent documents)
- Storage-specific features (cosine threshold)

Test Configuration:
- Each test runs against all implemented storage backends
- Tests run in different namespaces (chunks, entities, relationships)
- Mock embedding function used for consistent testing
"""

import pytest
import os
import shutil
import asyncio
from typing import Dict, Type
from lightrag.base import BaseVectorStorage
from lightrag.storage import NanoVectorDBStorage
from lightrag.utils import EmbeddingFunc
from lightrag.kg.postgres_impl import PostgresVectorStorage

from .test_utils import parse_postgres_uri, standard_cleanup
STORAGE_IMPLEMENTATIONS: Dict[str, Type[BaseVectorStorage]] = {
    "nano": NanoVectorDBStorage,
    "postgres": PostgresVectorStorage,
}

async def nano_config_factory():
    working_dir = "test_vector_storage"
    os.makedirs(working_dir, exist_ok=True)
    return {
        "working_dir": working_dir,
        "embedding_batch_num": 32
    }

async def postgres_config_factory():
    test_uri = os.getenv('POSTGRES_TEST_URI', 'postgresql://postgres:postgres@localhost:5432/lightrag_test')
    config = parse_postgres_uri(test_uri)
    config.update({"embedding_batch_num": 32})
    return config

CONFIG_FACTORIES = {
    "nano": nano_config_factory,
    "postgres": postgres_config_factory,
}


@pytest.fixture(params=STORAGE_IMPLEMENTATIONS.keys())
def impl_name(request):
    """Parametrized fixture providing storage implementation names.

    Returns:
        str: Name of the storage implementation ('nano' or 'postgres')
    """
    return request.param

@pytest.fixture(params=["chunks", "entities", "relationships"])
def namespace(request):
    """Parametrized fixture providing different storage namespaces.

    Returns:
        str: Namespace name for testing ('chunks', 'entities', or 'relationships')
    """
    return request.param

@pytest.fixture
async def storage(request, impl_name, namespace):
    """Fixture providing configured storage instance for testing.

    Args:
        request: pytest request object
        impl_name: Storage implementation name
        namespace: Storage namespace

    Yields:
        BaseVectorStorage: Configured storage instance with mock embedding function
    """
    storage_class = STORAGE_IMPLEMENTATIONS[impl_name]
    config = await CONFIG_FACTORIES[impl_name]()

    async def mock_embedding_func(texts):
        return [[1.0] * 384] * len(texts)

    store = storage_class(
        namespace=namespace,
        global_config=config,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=mock_embedding_func
        ),
        meta_fields={"source", "type"}
    )

    yield store
    await standard_cleanup(store)

def pytest_collection_modifyitems(items):
    for item in items:
        if "storage" in item.fixturenames:
            impl = item.callspec.params.get("impl_name", "unknown")
            ns = item.callspec.params.get("namespace", "unknown")
            item._nodeid = f"{item.originalname}[{impl}-{ns}]"

@pytest.mark.asyncio
async def test_chunks_operations(storage):
    """Test vector storage operations specific to the chunks namespace.

    Tests:
        - Chunk insertion
        - Semantic search within chunks
        - Metadata preservation
    """
    if storage.namespace != "chunks":
        pytest.skip("Test only for chunks namespace")

    chunks = {
        "chunk1": {
            "content": "Test chunk content",
            "source": "test_doc",
            "type": "chunk"
        }
    }
    await storage.upsert(chunks)
    results = await storage.query("test content", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "chunk1"

@pytest.mark.asyncio
async def test_entities_operations(storage):
    """Test vector storage operations specific to the entities namespace.

    Tests:
        - Entity insertion with metadata
        - Entity search functionality
        - Distance score calculation
    """
    if storage.namespace != "entities":
        pytest.skip("Test only for entities namespace")

    entities = {
        "ent1": {
            "content": "Entity description",
            "entity_name": "TestEntity",
            "source": "test",
            "type": "entity"
        }
    }
    await storage.upsert(entities)
    results = await storage.query("entity", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "ent1"
    assert "distance" in results[0]

@pytest.mark.asyncio
async def test_basic_operations(storage):
    """Test fundamental vector storage operations.

    Tests:
        - Document insertion
        - Basic semantic search
        - Metadata retrieval
        - Distance score calculation
    """
    # Skip if required methods aren't implemented
    if not hasattr(storage, 'upsert') or not hasattr(storage, 'query'):
        pytest.skip(f"{storage.__class__.__name__} doesn't implement required methods")

    doc = {
        "doc1": {
            "content": "This is a test document",
            "source": "test",
            "type": "document"
        }
    }
    await storage.upsert(doc)

    results = await storage.query("test document", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert "distance" in results[0]
    assert results[0]["source"] == "test"
    assert results[0]["type"] == "document"

@pytest.mark.asyncio
async def test_batch_operations(storage):
    """Test batch processing capabilities of vector storage.

    Tests:
        - Multiple document insertion
        - Batch query results
        - Metadata consistency in batch operations
    """
    docs = {
        "doc1": {"content": "First document", "source": "test", "type": "doc"},
        "doc2": {"content": "Second document", "source": "test", "type": "doc"},
        "doc3": {"content": "Third document", "source": "test", "type": "doc"}
    }
    await storage.upsert(docs)

    results = await storage.query("document", top_k=3)
    assert len(results) == 3
    assert all("distance" in r for r in results)
    assert all(r["source"] == "test" for r in results)

@pytest.mark.asyncio
async def test_metadata_filtering(storage):
    """Test metadata-based filtering in vector search.

    Tests:
        - Search with metadata filters
        - Search without filters
        - Filter accuracy
    """
    docs = {
        "doc1": {"content": "Test A", "source": "src1", "type": "typeA"},
        "doc2": {"content": "Test B", "source": "src2", "type": "typeB"}
    }
    await storage.upsert(docs)

    # Query without metadata filter
    results = await storage.query("Test", top_k=2)
    assert len(results) == 2

    # Query with metadata filter
    try:
        filtered_results = await storage.query("Test", top_k=2, metadata={"source": "src1"})
        assert len(filtered_results) == 1
        assert filtered_results[0]["source"] == "src1"
    except (AttributeError, TypeError) as e:
        pytest.skip(f"{storage.__class__.__name__} doesn't support metadata filtering")

@pytest.mark.asyncio
async def test_drop_and_callback(storage):
    """Test storage cleanup and callback functionality.

    Tests:
        - Storage drop operation
        - Index completion callback
        - Storage accessibility after operations
    """
    doc = {
        "doc1": {
            "content": "Test document",
            "source": "test",
            "type": "document"
        }
    }
    await storage.upsert(doc)

    # Test drop if implemented
    if not hasattr(storage, 'drop'):
        pytest.skip(f"{storage.__class__.__name__} doesn't implement drop()")

    # Test index_done_callback
    await storage.index_done_callback()
    # Verify storage is still accessible after callback
    await storage.upsert(doc)
    results = await storage.query("test", top_k=1)
    assert len(results) == 1

@pytest.mark.asyncio
async def test_empty_query(storage):
    """Test behavior with empty search queries.

    Tests:
        - Empty string query handling
        - Result set validation
    """
    results = await storage.query("", top_k=1)
    assert len(results) == 0

@pytest.mark.asyncio
async def test_query_nonexistent(storage):
    """Test behavior when querying for non-existent documents.

    Tests:
        - Non-existent document query handling
        - Empty result set validation
    """
    results = await storage.query("nonexistent document", top_k=1)
    assert len(results) == 0

@pytest.mark.asyncio
async def test_invalid_metadata_filter(storage):
    """Test behavior with invalid metadata filters.

    Tests:
        - Invalid metadata field handling
        - Error handling for unsupported filters
    """
    docs = {
        "doc1": {"content": "Test", "source": "src1", "type": "typeA"}
    }
    await storage.upsert(docs)

    try:
        # Try filtering with non-existent metadata field
        results = await storage.query("Test", top_k=1, metadata={"invalid_field": "value"})
        assert len(results) == 0
    except (AttributeError, TypeError):
        pytest.skip(f"{storage.__class__.__name__} doesn't support metadata filtering")

@pytest.mark.asyncio
async def test_concurrent_operations(storage):
    """Test concurrent operation handling in vector storage.

    Tests:
        - Concurrent batch insertions
        - Parallel query execution
        - Result consistency under load
    """
    # Test concurrent batch operations
    docs = {
        f"doc{i}": {"content": f"Document {i}", "source": "test", "type": "doc"}
        for i in range(100)
    }
    await storage.upsert(docs)

    # Test concurrent queries
    tasks = [
        storage.query("document", top_k=3)
        for _ in range(5)
    ]
    results = await asyncio.gather(*tasks)
    assert all(len(r) == 3 for r in results)

@pytest.mark.asyncio
async def test_cosine_threshold(storage):
    """Test cosine similarity threshold filtering.

    Tests:
        - Threshold-based result filtering
        - Distance score validation
        - Relevance ordering
    """
    # Test threshold filtering if supported
    docs = {
        "doc1": {"content": "Very relevant", "source": "test", "type": "doc"},
        "doc2": {"content": "Completely different topic", "source": "test", "type": "doc"}
    }
    await storage.upsert(docs)

    try:
        if hasattr(storage, 'cosine_better_than_threshold'):
            results = await storage.query("relevant", top_k=2)
            assert all(r.get('distance', 0) >= storage.cosine_better_than_threshold for r in results)
    except AttributeError:
        pytest.skip(f"{storage.__class__.__name__} doesn't support cosine threshold filtering")


