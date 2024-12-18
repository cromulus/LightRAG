import pytest
import os
import shutil
import asyncio
from typing import Dict, Type
from lightrag.base import BaseVectorStorage
from lightrag.storage import NanoVectorDBStorage
from lightrag.utils import EmbeddingFunc
from lightrag.kg.postgres_impl import PostgresVectorStorage

from .test_utils import parse_postgres_uri
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

async def nano_cleanup(store):
    working_dir = store.global_config["working_dir"]
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)

async def postgres_cleanup(store):
    # Clean up the test database tables
    await store.drop()
    await store.close()

CLEANUP_HANDLERS = {
    "nano": nano_cleanup,
    "postgres": postgres_cleanup,
}

async def nano_setup(store):
    pass

async def postgres_setup(store):
    await store.drop()


SETUP_HANDLERS = {
    "nano": nano_setup,
    "postgres": postgres_setup,
}

@pytest.fixture(params=STORAGE_IMPLEMENTATIONS.keys())
def impl_name(request):
    return request.param

@pytest.fixture(params=["chunks", "entities", "relationships"])
def namespace(request):
    return request.param

@pytest.fixture
async def storage(request, impl_name, namespace):
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
async def test_chunks_operations(storage):
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
    results = await storage.query("", top_k=1)
    assert len(results) == 0

@pytest.mark.asyncio
async def test_query_nonexistent(storage):
    results = await storage.query("nonexistent document", top_k=1)
    assert len(results) == 0

@pytest.mark.asyncio
async def test_invalid_metadata_filter(storage):
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


