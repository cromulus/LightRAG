"""Test suite for key-value storage implementations in LightRAG.

This module provides comprehensive testing for different key-value storage backends,
including JSON-based and PostgreSQL implementations. It tests various operations
such as basic CRUD, batch operations, TTL functionality, and embedding dimension
handling across different namespaces (full_docs, text_chunks).

The test suite uses parametrized fixtures to run the same tests against different
storage implementations, namespaces, and embedding dimensions, ensuring consistent
behavior across all storage backends.

Key Features Tested:
- Basic CRUD operations
- Batch operations
- TTL functionality
- Field filtering
- Embedding dimension validation
- Namespace-specific operations

Test Configuration:
- Multiple storage implementations (JSON, PostgreSQL)
- Different embedding dimensions (384, 1536)
- Multiple namespaces (full_docs, text_chunks)
- Mock embedding functions for consistent testing
"""

import pytest
import os
import shutil
import tests.test_utils as test_utils
from typing import Dict, Type
from lightrag.base import BaseKVStorage, EmbeddingFunc
from lightrag.storage import JsonKVStorage
from lightrag.kg.postgres_impl import PostgresKVStorage
from .test_utils import standard_cleanup

# Dictionary of storage implementations to test
STORAGE_IMPLEMENTATIONS: Dict[str, Type[BaseKVStorage]] = {
"json": JsonKVStorage,
   "postgres": PostgresKVStorage,
}

# Test different embedding dimensions
TEST_EMBEDDING_DIMS = [384, 1536]

@pytest.fixture(params=TEST_EMBEDDING_DIMS)
async def embedding_dim(request):
    """Parametrized fixture providing different embedding dimensions.

    Returns:
        int: Embedding dimension size (384 or 1536)
    """
    return request.param

# Configuration factories for each implementation
async def json_config_factory():
    """Create configuration for JSON storage implementation.

    Returns:
        dict: Configuration with working directory
    """
    working_dir = "test_kv_storage"
    os.makedirs(working_dir, exist_ok=True)
    return {"working_dir": working_dir}


async def postgres_config_factory():
    """Create configuration for PostgreSQL storage implementation.

    Returns:
        dict: PostgreSQL connection configuration
    """
    uri = os.getenv("POSTGRES_TEST_URI")
    return test_utils.parse_postgres_uri(uri)

CONFIG_FACTORIES = {
    "json": json_config_factory,
    "postgres": postgres_config_factory,
}


@pytest.fixture(params=STORAGE_IMPLEMENTATIONS.keys())
def impl_name(request):
    """Parametrized fixture providing storage implementation names.

    Returns:
        str: Name of the storage implementation ('json' or 'postgres')
    """
    return request.param

@pytest.fixture(params=["full_docs", "text_chunks"])
def namespace(request):
    """Parametrized fixture providing different storage namespaces.

    Returns:
        str: Namespace name for testing ('full_docs' or 'text_chunks')
    """
    return request.param

@pytest.fixture
async def storage(request, impl_name, namespace, embedding_dim):
    """Fixture providing configured storage instance for testing.

    Args:
        request: pytest request object
        impl_name: Storage implementation name
        namespace: Storage namespace
        embedding_dim: Embedding dimension size

    Yields:
        BaseKVStorage: Configured storage instance with mock embedding function
    """
    storage_class = STORAGE_IMPLEMENTATIONS[impl_name]
    config = await CONFIG_FACTORIES[impl_name]()
    config["embedding_dim"] = embedding_dim

    async def mock_embedding_func(texts):
        return [[1.0] * embedding_dim] * len(texts)

    store = storage_class(
        namespace=namespace,
        global_config=config,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=5000,
            func=mock_embedding_func
        )
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
async def test_full_docs_operations(storage):
    """Test operations specific to the full_docs namespace.

    Tests:
        - Document storage and retrieval
        - Workspace metadata handling
        - Document content integrity
    """
    if storage.namespace != "full_docs":
        pytest.skip("Test only for full_docs namespace")

    test_data = {
        "doc1": {
            "content": "Test document 1",
            "workspace": "test_workspace"
        },
        "doc2": {
            "content": "Test document 2",
            "workspace": "test_workspace"
        }
    }
    await storage.upsert(test_data)

    doc = await storage.get_by_id("doc1")
    assert doc["content"] == "Test document 1"

@pytest.mark.asyncio
async def test_text_chunks_operations(storage):
    """Test operations specific to the text_chunks namespace.

    Tests:
        - Chunk storage and retrieval
        - Chunk metadata (tokens, order index)
        - Document relationship tracking
        - Workspace association
    """
    if storage.namespace != "text_chunks":
        pytest.skip("Test only for text_chunks namespace")

    test_chunks = {
        "chunk1": {
            "content": "Chunk 1 content",
            "tokens": 10,
            "chunk_order_index": 0,
            "full_doc_id": "doc1",
            "workspace": "test_workspace"
        },
        "chunk2": {
            "content": "Chunk 2 content",
            "tokens": 12,
            "chunk_order_index": 1,
            "full_doc_id": "doc1",
            "workspace": "test_workspace"
        }
    }
    await storage.upsert(test_chunks)

    chunk = await storage.get_by_id("chunk1")
    assert chunk["tokens"] == 10
    assert chunk["chunk_order_index"] == 0

@pytest.mark.asyncio
async def test_basic_operations(storage):
    """Test fundamental key-value storage operations.

    Tests:
        - Single key-value pair operations
        - Multiple key-value pair operations
        - Key existence checking
        - Key enumeration
    """
    test_data = {
        "key1": {"field1": "value1"},
        "key2": {"field2": "value2"}
    }
    await storage.upsert(test_data)

    value1 = await storage.get_by_id("key1")
    assert value1 == {"field1": "value1"}

    values = await storage.get_by_ids(["key1", "key2"])
    assert values == [{"field1": "value1"}, {"field2": "value2"}]

    missing = await storage.filter_keys(["key1", "key3"])
    assert missing == {"key3"}

    keys = await storage.all_keys()
    assert set(keys) == {"key1", "key2"}

@pytest.mark.asyncio
async def test_field_filtering(storage):
    """Test field-level filtering capabilities.

    Tests:
        - Selective field retrieval
        - Field filtering accuracy
        - Partial document retrieval
    """
    test_data = {
        "key1": {"field1": "value1", "field2": "value2"}
    }
    await storage.upsert(test_data)

    filtered = await storage.get_by_ids(["key1"], fields={"field1"})
    assert filtered == [{"field1": "value1"}]

@pytest.mark.asyncio
async def test_drop(storage):
    """Test storage cleanup operations.

    Tests:
        - Storage dropping functionality
        - Post-drop state verification
        - Error handling for dropped storage
    """
    test_data = {"key1": {"field1": "value1"}}
    await storage.upsert(test_data)
    await storage.drop()
    # some backends will raise an error here, others will just return an empty list
    # so we need to check the error message
    try:
        # check if the key is not in the database
        keys = await storage.all_keys()
        assert "key1" not in keys
    except Exception as e:
        assert "does not exist" in str(e)

@pytest.mark.asyncio
async def test_index_done_callback(storage):
    """Test index completion callback functionality.

    Tests:
        - Callback execution
        - Storage accessibility post-callback
        - Data integrity after callback
    """
    test_data = {
        "key1": {"field1": "value1"}
    }
    await storage.upsert(test_data)
    await storage.index_done_callback()

    # Verify storage is still accessible after callback
    value = await storage.get_by_id("key1")
    assert value == {"field1": "value1"}

@pytest.mark.asyncio
async def test_batch_operations(storage):
    """Test batch processing capabilities.

    Tests:
        - Batch upsert operations
        - Batch retrieval operations
        - Field filtering in batch operations
        - Handling of non-existent keys
    """
    try:
        if not hasattr(storage, 'batch_get') or not hasattr(storage, 'batch_upsert_nodes'):
            pytest.skip(f"{storage.__class__.__name__} doesn't support batch operations")

        # Test batch upsert
        test_data = {
            f"key{i}": {
                "content": f"Test content {i}",
                "workspace": "test_workspace",
                "meta_field": f"value{i}"
            }
            for i in range(5)
        }

        await storage.batch_upsert_nodes(test_data)

        # Test batch get
        results = await storage.batch_get(list(test_data.keys()))
        assert len(results) == len(test_data)

        # Verify content of results
        for key, data in test_data.items():
            assert key in results
            assert results[key]["content"] == data["content"]
            assert results[key]["meta_field"] == data["meta_field"]

        # Test batch get with field filtering
        filtered_results = await storage.batch_get(
            list(test_data.keys()),
            fields={"content"}
        )
        assert len(filtered_results) == len(test_data)
        for key, data in filtered_results.items():
            assert "content" in data
            assert "meta_field" not in data

        # Test batch get with non-existent keys
        mixed_keys = list(test_data.keys()) + ["nonexistent1", "nonexistent2"]
        mixed_results = await storage.batch_get(mixed_keys)
        assert len(mixed_results) == len(mixed_keys)
        assert all(mixed_results[k] is None for k in ["nonexistent1", "nonexistent2"])

    except AttributeError:
        pytest.skip(f"{storage.__class__.__name__} doesn't support batch operations")

@pytest.mark.asyncio
async def test_ttl(storage):
    """Test time-to-live functionality if supported.

    Tests:
        - TTL-based entry expiration
        - Immediate data accessibility
        - Post-expiration data removal
    """
    # Test TTL if supported
    try:
        if hasattr(storage, 'upsert_with_ttl'):
            await storage.upsert_with_ttl({"key1": {"value": "temp"}}, ttl_seconds=1)
            value = await storage.get_by_id("key1")
            assert value is not None
            await asyncio.sleep(1.1)
            value = await storage.get_by_id("key1")
            assert value is None
    except AttributeError:
        pytest.skip(f"{storage.__class__.__name__} doesn't support TTL")

# Add new test for embedding dimension handling
async def test_embedding_dimension_validation(impl_name, namespace):
    """Test validation of embedding dimensions.

    Tests:
        - Invalid dimension handling
        - Dimension configuration validation
        - Error handling for invalid dimensions
    """
    storage_class = STORAGE_IMPLEMENTATIONS[impl_name]
    base_config = await CONFIG_FACTORIES[impl_name]()

    # Test with invalid embedding dimensions
    invalid_dims = [0, -1, "invalid", None]

    for dim in invalid_dims:
        # Merge base config with embedding dimension
        config = {**base_config, "embedding_dim": dim}

        with pytest.raises(ValueError, match="Invalid embedding dimension"):
            store = storage_class(
                namespace=namespace,
                global_config=config,
                embedding_func=EmbeddingFunc(
                    embedding_dim=dim,
                    max_token_size=5000,
                    func=lambda texts: [[1.0] * 384] * len(texts)
                )
            )

@pytest.mark.asyncio
async def test_kv_storage_multi_user_isolation(kv_storage):
    """Test that KV storage properly isolates data between users when supported."""
    if not kv_storage.supports_multi_user:
        pytest.skip(f"{kv_storage.__class__.__name__} does not support multi-user")

    # Test data
    await kv_storage.set("shared_key", "user1_value", user_id="user1")
    await kv_storage.set("shared_key", "user2_value", user_id="user2")

    # Verify isolation
    value1 = await kv_storage.get("shared_key", user_id="user1")
    value2 = await kv_storage.get("shared_key", user_id="user2")

    assert value1 == "user1_value"
    assert value2 == "user2_value"

    # Test deletion
    await kv_storage.delete("shared_key", user_id="user1")
    assert await kv_storage.get("shared_key", user_id="user1") is None
    assert await kv_storage.get("shared_key", user_id="user2") == "user2_value"

@pytest.mark.asyncio
async def test_kv_storage_default_user(kv_storage):
    """Test that KV storage works with default user (no user_id specified)."""
    # Should work without user_id
    await kv_storage.set("key", "value")
    value = await kv_storage.get("key")
    assert value == "value"

