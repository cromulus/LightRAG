import pytest
import os
import shutil
import numpy as np
from typing import Type
from lightrag.base import BaseVectorStorage
from lightrag.storage import NanoVectorDBStorage
from lightrag.kg.postgres_impl import PostgresVectorDBStorage
from lightrag.utils import EmbeddingFunc

async def mock_embedding_func(texts):
    """Mock embedding function that returns fixed-size vectors"""
    return np.array([[1.0] * 384] * len(texts))

class BaseVectorStorageTest:
    """Base test class for vector storage implementations"""

    storage_class: Type[BaseVectorStorage] = None

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
                func=mock_embedding_func
            ),
            meta_fields={"source", "type"}
        )

        yield storage
        await storage.drop()
        if hasattr(storage, 'pool') and storage.pool:
            await storage.pool.close()

    @pytest.mark.asyncio
    async def test_basic_operations(self, storage):
        """Test basic vector storage operations"""
        # Test single document insertion
        doc = {
            "doc1": {
                "content": "This is a test document",
                "source": "test",
                "type": "document"
            }
        }
        await storage.upsert(doc)

        # Test vector search
        results = await storage.query("test document", top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert "distance" in results[0]
        assert results[0]["source"] == "test"
        assert results[0]["type"] == "document"

    @pytest.mark.asyncio
    async def test_batch_operations(self, storage):
        """Test batch operations"""
        docs = {
            f"doc{i}": {
                "content": f"Test document {i}",
                "source": "test",
                "type": "document"
            }
            for i in range(3)
        }
        await storage.upsert(docs)

        results = await storage.query("test", top_k=3)
        assert len(results) == 3
        assert all("id" in r and r["id"].startswith("doc") for r in results)
        assert all("distance" in r for r in results)

    @pytest.mark.asyncio
    async def test_metadata_filtering(self, storage):
        """Test metadata handling"""
        docs = {
            "doc1": {
                "content": "Test document A",
                "source": "source_a",
                "type": "type_a"
            },
            "doc2": {
                "content": "Test document B",
                "source": "source_b",
                "type": "type_b"
            }
        }
        await storage.upsert(docs)

        results = await storage.query("test", top_k=2)
        assert len(results) == 2
        assert all("source" in r for r in results)
        assert all("type" in r for r in results)

    @pytest.mark.asyncio
    async def test_drop(self, storage):
        """Test drop functionality"""
        doc = {
            "doc1": {
                "content": "Test document",
                "source": "test",
                "type": "document"
            }
        }
        await storage.upsert(doc)
        await storage.drop()

        # After drop, query should return empty results
        results = await storage.query("test", top_k=1)
        assert len(results) == 0

class TestNanoVectorDBStorage(BaseVectorStorageTest):
    """Test NanoVectorDB implementation"""
    storage_class = NanoVectorDBStorage

    @classmethod
    def config_factory(cls):
        working_dir = "test_vector_storage"
        os.makedirs(working_dir, exist_ok=True)
        return {
            "working_dir": working_dir,
            "embedding_batch_num": 32
        }

    @pytest.fixture(autouse=True)
    async def cleanup_after_test(self):
        yield
        # Cleanup after each test
        working_dir = self.config_factory()["working_dir"]
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)

class TestPostgresVectorDBStorage(BaseVectorStorageTest):
    """Test PostgreSQL vector storage implementation"""
    storage_class = PostgresVectorDBStorage

    @classmethod
    def config_factory(cls):
        return {
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "user": "lightrag_test",
                "password": "lightrag_test",
                "database": "lightrag_test"
            },
            "embedding_batch_num": 32
        }

    @pytest.fixture(autouse=True)
    async def ensure_db_setup(self, storage):
        """Ensure test database is set up before tests"""
        from tests.setup_postgres_db import setup_postgres
        import asyncio
        await setup_postgres()
        await asyncio.sleep(1)
        yield

