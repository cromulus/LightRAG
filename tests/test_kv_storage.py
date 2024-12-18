import pytest
import os
import shutil
import tests.test_utils as test_utils
from typing import Dict, Type
from lightrag.base import BaseKVStorage, EmbeddingFunc
from lightrag.storage import JsonKVStorage
#from lightrag.kg.postgres_impl import PostgresKVStorage

# Dictionary of storage implementations to test
STORAGE_IMPLEMENTATIONS: Dict[str, Type[BaseKVStorage]] = {
    "json": JsonKVStorage,
#    "postgres": PostgresKVStorage,
}


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

async def json_setup(store):
    pass

async def postgres_setup(store):
    await store.drop() # drop all old tables
    await store.check_tables() # create new tables

SETUP_HANDLERS = {
    "json": json_setup,
    "postgres": postgres_setup,
}


# Cleanup handlers for each implementation
async def json_cleanup(store):
    working_dir = store.global_config["working_dir"]
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)

async def postgres_cleanup(store):
    await store.drop()
    await store.close()

CLEANUP_HANDLERS = {
    "json": json_cleanup,
    "postgres": postgres_cleanup,
}

@pytest.fixture(params=STORAGE_IMPLEMENTATIONS.keys())
def impl_name(request):
    return request.param

@pytest.fixture(params=["full_docs", "text_chunks"])
def namespace(request):
    return request.param

@pytest.fixture
async def storage(request, impl_name, namespace):
    storage_class = STORAGE_IMPLEMENTATIONS[impl_name]
    config = await CONFIG_FACTORIES[impl_name]()

    embedding_func = EmbeddingFunc(
        embedding_dim=384,
        max_token_size=5000,
        func=lambda texts: [[1.0] * 384] * len(texts)
    )

    store = storage_class(
        namespace=namespace,
        global_config=config,
        embedding_func=embedding_func
    )

    setup_handler = SETUP_HANDLERS[impl_name]
    await setup_handler(store)

    yield store

    cleanup_handler = CLEANUP_HANDLERS[impl_name]
    await cleanup_handler(store)

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

    keys = await storage.all_keys()
    assert len(keys) == 0

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
        if not hasattr(storage, 'batch_get'):
            pytest.skip(f"{storage.__class__.__name__} doesn't support batch operations")

        data = {
            f"key{i}": {"value": f"value{i}"}
            for i in range(5)
        }
        await storage.upsert(data)
        results = await storage.batch_get(list(data.keys()))
        assert len(results) == len(data)
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

