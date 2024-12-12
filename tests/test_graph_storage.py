import pytest
import os
import shutil
import tempfile
from typing import Type
from lightrag.base import BaseGraphStorage
from lightrag.storage import NetworkXStorage
from lightrag.kg.postgres_impl import PostgresGraphStorage
from lightrag.utils import EmbeddingFunc

class BaseGraphStorageTest:
    """Base test class for graph storage implementations"""

    storage_class: Type[BaseGraphStorage] = None

    @classmethod
    def config_factory(cls):
        """Override this method in subclasses to provide storage-specific configuration"""
        return {}

    @pytest.fixture
    async def storage_config(self):
        """Get storage configuration"""
        return self.config_factory()

    @pytest.fixture
    async def storage(self, storage_config):
        """Create a storage instance for testing"""
        assert self.storage_class is not None, "storage_class must be set in subclass"

        storage = self.storage_class(
            namespace="test",
            global_config=storage_config,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=5000,
                func=lambda texts: [[1.0] * 384] * len(texts)  # Mock embedding for testing
            )
        )

        yield storage
        await storage.drop()
        if hasattr(storage, 'pool'):
            await storage.close()

    @pytest.mark.asyncio
    async def test_basic_node_operations(self, storage):
        """Test basic node operations"""
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
    async def test_basic_edge_operations(self, storage):
        """Test basic edge operations"""
        await storage.upsert_node("node1", {"name": "Node 1"})
        await storage.upsert_node("node2", {"name": "Node 2"})

        edge_data = {
            "type": "test_relation",
            "weight": 1.0
        }
        await storage.upsert_edge("node1", "node2", edge_data)

        assert await storage.has_edge("node1", "node2")
        assert not await storage.has_edge("node1", "nonexistent")

        edge = await storage.get_edge("node1", "node2")
        assert edge["type"] == "test_relation"
        assert edge["weight"] == 1.0
        assert await storage.edge_degree("node1", "node2") > 0

    @pytest.mark.asyncio
    async def test_node_edges(self, storage):
        """Test node edge retrieval"""
        await storage.upsert_node("center", {"name": "Center"})
        nodes = ["node1", "node2", "node3"]

        for node in nodes:
            await storage.upsert_node(node, {"name": node})
            await storage.upsert_edge(node, "center", {"type": "connects"})

        edges = await storage.get_node_edges("center")
        assert len(edges) == len(nodes)
        connected = {edge[0] if edge[1] == "center" else edge[1] for edge in edges}
        assert all(node in connected for node in nodes)

    @pytest.mark.asyncio
    async def test_node_deletion(self, storage):
        """Test node deletion with connected edges"""
        await storage.upsert_node("node1", {"name": "Node 1"})
        await storage.upsert_node("node2", {"name": "Node 2"})
        await storage.upsert_edge("node1", "node2", {"type": "connects"})

        await storage.delete_node("node1")
        assert not await storage.has_node("node1")
        assert not await storage.has_edge("node1", "node2")

    @pytest.mark.asyncio
    async def test_drop(self, storage):
        """Test drop functionality"""
        await storage.upsert_node("node1", {"name": "Node 1"})
        await storage.upsert_node("node2", {"name": "Node 2"})
        await storage.upsert_edge("node1", "node2", {"type": "connects"})

        await storage.drop()
        assert not await storage.has_node("node1")
        assert not await storage.has_node("node2")
        assert not await storage.has_edge("node1", "node2")

class TestNetworkXStorage(BaseGraphStorageTest):
    """Test NetworkX implementation"""
    storage_class = NetworkXStorage

    @classmethod
    def config_factory(cls):
        working_dir = "test_graph_storage"
        os.makedirs(working_dir, exist_ok=True)
        return {"working_dir": working_dir}

    @pytest.fixture(autouse=True)
    async def cleanup_after_test(self):
        yield
        # Cleanup after each test
        working_dir = self.config_factory()["working_dir"]
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)

class TestPostgresGraphStorage(BaseGraphStorageTest):
    """Test PostgreSQL graph storage implementation"""
    storage_class = PostgresGraphStorage

    @classmethod
    def config_factory(cls):
        from tests.setup_postgres_db import parse_postgres_uri
        test_uri = os.getenv('POSTGRES_TEST_URI', 'postgresql://lightrag_test:lightrag_test@db:5432/lightrag_test')
        return {
            "postgres": parse_postgres_uri(test_uri)
        }

    @pytest.fixture(autouse=True)
    async def ensure_db_setup(self, storage):
        """Ensure test database is set up before tests"""
        from tests.setup_postgres_db import setup_postgres
        import asyncio
        await setup_postgres()
        await asyncio.sleep(1)
        yield
