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
    return request.param

# Configuration factories for each implementation
async def json_config_factory():
    working_dir = "test_kv_storage"
    os.makedirs(working_dir, exist_ok=True)
    return {"working_dir": working_dir}


async def postgres_config_factory():
    uri = os.getenv("POSTGRES_TEST_URI")
    return test_utils.parse_postgres_uri(uri)

CONFIG_FACTORIES = {
    "json": json_config_factory,
    "postgres": postgres_config_factory,
}


@pytest.fixture(params=STORAGE_IMPLEMENTATIONS.keys())
def impl_name(request):
    return request.param

@pytest.fixture(params=["full_docs", "text_chunks"])
def namespace(request):
    return request.param

@pytest.fixture
async def storage(request, impl_name, namespace, embedding_dim):
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
    test_data = {
        "key1": {"field1": "value1", "field2": "value2"}
    }
    await storage.upsert(test_data)

    filtered = await storage.get_by_ids(["key1"], fields={"field1"})
    assert filtered == [{"field1": "value1"}]

@pytest.mark.asyncio
async def test_drop(storage):
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

