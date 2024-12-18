import pytest
import os
import shutil
from typing import Dict, Type
from lightrag.base import BaseGraphStorage
from lightrag.storage import NetworkXStorage
from lightrag.utils import EmbeddingFunc
from lightrag.kg.age_impl import AGEStorage
from tests.test_utils import parse_postgres_uri
import psycopg
from psycopg.errors import DuplicateSchema, InvalidParameterValue

STORAGE_IMPLEMENTATIONS: Dict[str, Type[BaseGraphStorage]] = {
    "networkx": NetworkXStorage,
    "age": AGEStorage,
}

def networkx_config_factory():
    working_dir = "test_graph_storage"
    os.makedirs(working_dir, exist_ok=True)
    return {"working_dir": working_dir}


def age_config_factory():
    # Use POSTGRES_TEST_URI if available
    test_uri = os.getenv('POSTGRES_TEST_URI', 'postgresql://postgres:postgres@localhost:5432/lightrag_test')
    config = parse_postgres_uri(test_uri)

    return {
        "age": {
            "graph_name": "test_graph",
            "host": config.get("host", "localhost"),
            "port": config.get("port", 5432),
            "database": config.get("database", "lightrag_test"),
            "user": config.get("user", "postgres"),
            "password": config.get("password", "postgres"),
            "search_path": "ag_catalog, public",  # Default search path
            # "search_path": "ag_catalog, \"$user\", public"  # Optional custom search path
        }
    }

CONFIG_FACTORIES = {
    "networkx": networkx_config_factory,
    "age": age_config_factory,
}


async def age_setup(store):
    await store.drop()


async def networkx_setup(store):
    pass

SETUP_HANDLERS = {
    "networkx": networkx_setup,
    "age": age_setup,
}

async def age_cleanup(store):
    try:
        await store.drop()
    finally:
        await store.close()

async def networkx_cleanup(store):
    working_dir = store.global_config["working_dir"]
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)

CLEANUP_HANDLERS = {
    "networkx": networkx_cleanup,
    "age": age_cleanup,
}

@pytest.fixture(params=STORAGE_IMPLEMENTATIONS.keys())
def impl_name(request):
    return request.param

@pytest.fixture
async def storage(impl_name):
    storage_class = STORAGE_IMPLEMENTATIONS[impl_name]
    config = CONFIG_FACTORIES[impl_name]()

    async def mock_embedding_func(texts):
        return [[1.0] * 384] * len(texts)

    store = storage_class(
        namespace="test",
        global_config=config,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=mock_embedding_func
        ),
    )

    try:
        if impl_name in SETUP_HANDLERS:
            await SETUP_HANDLERS[impl_name](store)
        yield store
    finally:
        if hasattr(store, 'close'):
            await store.close()

@pytest.mark.asyncio
async def test_basic_node_operations(storage):
    node_data = {
        "name": "Test Node",
        "type": "test",
        "description": "A test node"
    }
    await storage.upsert_node("node1", node_data)

    assert await storage.has_node("node1")
    assert not await storage.has_node("nonexistent")

    node = await storage.get_node("node1")
    assert node["name"] == "Test Node"
    assert node["type"] == "test"
    assert await storage.node_degree("node1") == 0

@pytest.mark.asyncio
async def test_basic_edge_operations(storage):
    src_id = "source_node"
    tgt_id = "target_node"
    edge_data = {"weight": 1.0}

    await storage.upsert_node(src_id, {})
    await storage.upsert_node(tgt_id, {})

    await storage.upsert_edge(src_id, tgt_id, edge_data)
    assert await storage.has_edge(src_id, tgt_id)

    edge = await storage.get_edge(src_id, tgt_id)
    assert edge["weight"] == 1.0

@pytest.mark.asyncio
async def test_node_edges(storage):
    src_id = "source"
    tgt_ids = ["target1", "target2"]

    await storage.upsert_node(src_id, {})
    for tgt_id in tgt_ids:
        await storage.upsert_node(tgt_id, {})
        await storage.upsert_edge(src_id, tgt_id, {})

    edges = await storage.get_node_edges(src_id)
    assert len(edges) == len(tgt_ids)

@pytest.mark.asyncio
async def test_drop_and_callback(storage):
    node_data = {"name": "Test Node"}
    await storage.upsert_node("node1", node_data)

    # Test drop if implemented
    if hasattr(storage, 'drop'):
        await storage.drop()
        assert not await storage.has_node("node1")

    # Test index_done_callback
    await storage.index_done_callback()
    await storage.upsert_node("node2", node_data)
    assert await storage.has_node("node2")

@pytest.mark.asyncio
async def test_delete_node(storage):
    node_data = {"name": "Test Node"}
    await storage.upsert_node("node1", node_data)
    assert await storage.has_node("node1")

    await storage.delete_node("node1")
    assert not await storage.has_node("node1")

@pytest.mark.asyncio
async def test_node_embedding(storage):
    try:
        from graspologic import embed
    except ImportError:
        pytest.skip(f"graspologic not installed, {storage.__class__.__name__} node embedding test skipped")

    nodes = {
        "node1": {"name": "Node 1"},
        "node2": {"name": "Node 2"}
    }
    for node_id, data in nodes.items():
        await storage.upsert_node(node_id, data)
    await storage.upsert_edge("node1", "node2", {"type": "test"})

    try:
        embeddings, node_ids = await storage.embed_nodes("node2vec")
        assert len(embeddings) == len(node_ids)
        assert all(isinstance(e, np.ndarray) for e in embeddings)
    except (NotImplementedError, ValueError):
        pytest.skip(f"{storage.__class__.__name__} doesn't support node2vec embedding")

@pytest.mark.asyncio
async def test_graph_statistics(storage):
    # Test statistics methods if implemented
    try:
        if hasattr(storage, 'get_statistics'):
            stats = await storage.get_statistics()
            assert isinstance(stats, dict)
            assert 'nodes_count' in stats
            assert 'edges_count' in stats
    except AttributeError:
        pytest.skip(f"{storage.__class__.__name__} doesn't support statistics")

@pytest.mark.asyncio
async def test_batch_operations(storage):
    # Test batch node/edge operations if supported
    nodes = {
        f"node{i}": {"name": f"Node {i}"}
        for i in range(5)
    }
    try:
        if hasattr(storage, 'batch_upsert_nodes'):
            await storage.batch_upsert_nodes(nodes)
            for node_id in nodes:
                assert await storage.has_node(node_id)
    except AttributeError:
        pytest.skip(f"{storage.__class__.__name__} doesn't support batch operations")

