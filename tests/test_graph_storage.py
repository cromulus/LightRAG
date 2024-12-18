"""Test suite for graph storage implementations in LightRAG.

This module provides comprehensive testing for different graph storage backends,
including NetworkX and Apache AGE implementations. It tests various graph operations
such as node/edge management, embeddings, and graph statistics.

The test suite uses parametrized fixtures to run the same tests against different
storage implementations, ensuring consistent behavior across all graph backends.

Key Features Tested:
- Basic node and edge operations
- Graph topology operations
- Node embeddings
- Graph statistics
- Batch operations
- Storage cleanup and callbacks

Test Configuration:
- Multiple storage implementations (NetworkX, Apache AGE)
- Mock embedding functions for consistent testing
- Configurable graph storage backends
- Optional features testing (embeddings, statistics)
"""

import pytest
import os
import shutil
from typing import Dict, Type
from lightrag.base import BaseGraphStorage
from lightrag.storage import NetworkXStorage
from lightrag.utils import EmbeddingFunc
from lightrag.kg.age_impl import AGEStorage
from tests.test_utils import parse_postgres_uri, standard_cleanup

STORAGE_IMPLEMENTATIONS: Dict[str, Type[BaseGraphStorage]] = {
    "networkx": NetworkXStorage,
    "age": AGEStorage,
}

def networkx_config_factory():
    """Create configuration for NetworkX storage implementation.

    Returns:
        dict: Configuration with working directory
    """
    working_dir = "test_graph_storage"
    os.makedirs(working_dir, exist_ok=True)
    return {"working_dir": working_dir}

def age_config_factory():
    """Create configuration for Apache AGE storage implementation.

    Returns:
        dict: AGE connection configuration with graph settings
    """
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
            "search_path": "ag_catalog, public",
        }
    }

CONFIG_FACTORIES = {
    "networkx": networkx_config_factory,
    "age": age_config_factory,
}

@pytest.fixture(params=STORAGE_IMPLEMENTATIONS.keys())
def impl_name(request):
    """Parametrized fixture providing storage implementation names.

    Returns:
        str: Name of the storage implementation ('networkx' or 'age')
    """
    return request.param

@pytest.fixture
async def storage(impl_name):
    """Fixture providing configured graph storage instance for testing.

    Args:
        impl_name: Storage implementation name

    Yields:
        BaseGraphStorage: Configured storage instance with mock embedding function
    """
    storage_class = STORAGE_IMPLEMENTATIONS[impl_name]
    config = CONFIG_FACTORIES[impl_name]()

    store = storage_class(
        namespace="test",
        global_config=config,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=lambda texts: [[1.0] * 384] * len(texts)
        )
    )

    yield store
    await standard_cleanup(store)

@pytest.mark.asyncio
async def test_basic_node_operations(storage):
    """Test fundamental node operations in graph storage.

    Tests:
        - Node creation and update
        - Node existence checking
        - Node data retrieval
        - Node degree calculation
    """
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
    """Test fundamental edge operations in graph storage.

    Tests:
        - Edge creation between nodes
        - Edge existence checking
        - Edge data retrieval
        - Edge properties management
    """
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
    """Test node edge relationship operations.

    Tests:
        - Multiple edge creation
        - Edge listing for nodes
        - Edge count verification
    """
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
    """Test storage cleanup and callback functionality.

    Tests:
        - Storage dropping
        - Post-drop state verification
        - Index completion callback
        - Storage accessibility after operations
    """
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
    """Test node deletion operations.

    Tests:
        - Node removal
        - Post-deletion state verification
        - Node existence checking
    """
    node_data = {"name": "Test Node"}
    await storage.upsert_node("node1", node_data)
    assert await storage.has_node("node1")

    await storage.delete_node("node1")
    assert not await storage.has_node("node1")

@pytest.mark.asyncio
async def test_node_embedding(storage):
    """Test node embedding generation capabilities.

    Tests:
        - Node2vec embedding generation
        - Embedding dimensionality
        - Embedding validity
        - Optional feature handling
    """
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
    """Test graph statistics collection functionality.

    Tests:
        - Basic graph metrics
        - Node and edge counts
        - Optional statistics features
    """
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
    """Test batch processing capabilities.

    Tests:
        - Batch node creation
        - Batch operation validation
        - Optional feature handling
    """
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

