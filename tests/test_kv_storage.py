import pytest
import os
import shutil
from typing import Type, Dict, Any
from lightrag.base import BaseKVStorage
from lightrag.storage import JsonKVStorage
from lightrag.kg.postgres_impl import PostgresKVStorage

def dummy_embedding_func(text):
    """Dummy embedding function that returns a fixed vector"""
    return [0.1] * 10

class BaseKVStorageTest:
    """Base test class for KV storage implementations"""

    storage_class: Type[BaseKVStorage] = None

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

        # Create test working directory
        working_dir = "test_kv_storage"
        os.makedirs(working_dir, exist_ok=True)

        # Initialize storage
        storage = self.storage_class(
            namespace="test",
            global_config={"working_dir": working_dir, **storage_config},
            embedding_func=dummy_embedding_func
        )

        yield storage

        # Cleanup
        await storage.drop()
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)

    @pytest.mark.asyncio
    async def test_basic_operations(self, storage):
        """Test basic CRUD operations"""
        # Test empty storage
        keys = await storage.all_keys()
        assert len(keys) == 0

        # Test single item operations
        test_data = {"key1": {"field1": "value1"}}
        await storage.upsert(test_data)

        # Test get_by_id
        value = await storage.get_by_id("key1")
        assert value == {"field1": "value1"}

        # Test non-existent key
        value = await storage.get_by_id("nonexistent")
        assert value is None

        # Test all_keys
        keys = await storage.all_keys()
        assert set(keys) == {"key1"}

    @pytest.mark.asyncio
    async def test_batch_operations(self, storage):
        """Test batch operations"""
        # Test batch upsert
        test_data = {
            "key1": {"field1": "value1"},
            "key2": {"field2": "value2"}
        }
        await storage.upsert(test_data)

        # Test get_by_ids
        values = await storage.get_by_ids(["key1", "key2"])
        assert len(values) == 2
        assert values[0] == {"field1": "value1"}
        assert values[1] == {"field2": "value2"}

        # Test get_by_ids with non-existent keys
        values = await storage.get_by_ids(["key1", "nonexistent", "key2"])
        assert len(values) == 3
        assert values[0] == {"field1": "value1"}
        assert values[1] is None
        assert values[2] == {"field2": "value2"}

    @pytest.mark.asyncio
    async def test_field_filtering(self, storage):
        """Test field filtering in get_by_ids"""
        # Insert test data
        test_data = {
            "key1": {"field1": "value1", "field2": "value2"},
            "key2": {"field1": "value3", "field2": "value4"}
        }
        await storage.upsert(test_data)

        # Test field filtering
        values = await storage.get_by_ids(["key1", "key2"], fields={"field1"})
        assert len(values) == 2
        assert values[0] == {"field1": "value1"}
        assert values[1] == {"field1": "value3"}

    @pytest.mark.asyncio
    async def test_filter_keys(self, storage):
        """Test filter_keys functionality"""
        # Insert test data
        test_data = {
            "key1": {"field1": "value1"},
            "key2": {"field2": "value2"}
        }
        await storage.upsert(test_data)

        # Test filtering existing and non-existing keys
        keys_to_filter = ["key1", "nonexistent1", "key2", "nonexistent2"]
        non_existent = await storage.filter_keys(keys_to_filter)
        assert non_existent == {"nonexistent1", "nonexistent2"}

    @pytest.mark.asyncio
    async def test_drop(self, storage):
        """Test drop functionality"""
        # Insert test data
        test_data = {
            "key1": {"field1": "value1"},
            "key2": {"field2": "value2"}
        }
        await storage.upsert(test_data)

        # Verify data is inserted
        keys = await storage.all_keys()
        assert len(keys) == 2

        # Drop all data
        await storage.drop()

        # Verify storage is empty
        keys = await storage.all_keys()
        assert len(keys) == 0

class TestJsonKVStorage(BaseKVStorageTest):
    """Test JsonKVStorage implementation"""
    storage_class = JsonKVStorage

class TestPostgresKVStorage(BaseKVStorageTest):
    """Test PostgresKVStorage implementation"""
    storage_class = PostgresKVStorage

    @classmethod
    def config_factory(cls):
        """Create PostgreSQL configuration for testing"""
        return {
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "user": "lightrag_test",
                "password": "lightrag_test",
                "database": "lightrag_test"
            }
        }

    @pytest.fixture(autouse=True)
    async def setup_postgres(self):
        """Set up PostgreSQL database before tests"""
        from tests.setup_postgres_db import setup_postgres
        await setup_postgres()
        yield
