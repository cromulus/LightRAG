import pytest
import asyncio
from lightrag.kg.postgres_impl import PostgresKVStorage

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
async def kv_store(pg_config):
    store = PostgresKVStorage(
        namespace="test",
        global_config=pg_config,
        embedding_func=None
    )
    yield store
    await store.drop()
    await store.close()

@pytest.mark.asyncio
async def test_basic_operations(kv_store):
    # Test upsert and get
    test_data = {
        "key1": {"field1": "value1"},
        "key2": {"field2": "value2"}
    }
    await kv_store.upsert(test_data)

    # Test get_by_id
    value1 = await kv_store.get_by_id("key1")
    assert value1 == {"field1": "value1"}

    # Test get_by_ids
    values = await kv_store.get_by_ids(["key1", "key2"])
    assert values == [{"field1": "value1"}, {"field2": "value2"}]

    # Test filter_keys
    missing = await kv_store.filter_keys(["key1", "key3"])
    assert missing == {"key3"}

    # Test all_keys
    keys = await kv_store.all_keys()
    assert set(keys) == {"key1", "key2"}

@pytest.mark.asyncio
async def test_field_filtering(kv_store):
    test_data = {
        "key1": {"field1": "value1", "field2": "value2"}
    }
    await kv_store.upsert(test_data)

    filtered = await kv_store.get_by_ids(["key1"], fields={"field1"})
    assert filtered == [{"field1": "value1"}]

@pytest.mark.asyncio
async def test_drop(kv_store):
    test_data = {"key1": {"field1": "value1"}}
    await kv_store.upsert(test_data)
    await kv_store.drop()

    keys = await kv_store.all_keys()
    assert len(keys) == 0
