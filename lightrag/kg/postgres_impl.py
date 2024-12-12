import asyncpg
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Union
import json

from ..base import BaseKVStorage, BaseVectorStorage, BaseGraphStorage
from ..utils import logger

@dataclass
class PostgresKVStorage(BaseKVStorage):
    """PostgreSQL-based key-value storage using JSONB"""

    pool: Optional[asyncpg.Pool] = None
    _pool_lock = None  # Will be initialized in __post_init__

    def __post_init__(self):
        """Initialize connection pool and create table if needed"""
        import asyncio
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool with retry logic"""
        if self.pool is None:
            async with self._pool_lock:
                if self.pool is None:  # Double check under lock
                    try:
                        self.pool = await asyncpg.create_pool(**self.global_config["postgres"])
                        await self._init_table()
                    except Exception as e:
                        logger.error(f"Failed to initialize PostgreSQL connection: {e}")
                        raise
        return self.pool

    async def _init_table(self):
        """Create the KV store table if it doesn't exist"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS kv_store (
                    namespace TEXT,
                    key TEXT,
                    value JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (namespace, key)
                );

                CREATE INDEX IF NOT EXISTS idx_kv_store_namespace
                ON kv_store(namespace);
            ''')
            logger.info(f"Initialized PostgreSQL KV store table for namespace '{self.namespace}'")

    async def all_keys(self) -> List[str]:
        """Get all keys for the current namespace"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT key FROM kv_store WHERE namespace = $1;
            ''', self.namespace)
            return [row['key'] for row in rows]

    async def get_by_id(self, id: str) -> Union[Dict, None]:
        """Get value by key"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT value FROM kv_store
                WHERE namespace = $1 AND key = $2;
            ''', self.namespace, id)
            if not row:
                return None
            return json.loads(row['value']) if isinstance(row['value'], str) else row['value']

    async def get_by_ids(
        self,
        ids: List[str],
        fields: Optional[Set[str]] = None
    ) -> List[Union[Dict, None]]:
        """Get multiple values by keys with optional field filtering"""
        if not ids:
            return []

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT key, value FROM kv_store
                WHERE namespace = $1 AND key = ANY($2)
                ORDER BY array_position($2::text[], key);
            ''', self.namespace, ids)

            result = {}
            for row in rows:
                value = json.loads(row['value']) if isinstance(row['value'], str) else row['value']
                if fields:
                    value = {k: v for k, v in value.items() if k in fields}
                result[row['key']] = value

            return [result.get(id) for id in ids]

    async def filter_keys(self, data: List[str]) -> Set[str]:
        """Find keys that don't exist in storage"""
        if not data:
            return set()

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT key FROM kv_store
                WHERE namespace = $1 AND key = ANY($2);
            ''', self.namespace, data)
            existing_keys = {row['key'] for row in rows}
            return set(data) - existing_keys

    async def upsert(self, data: Dict[str, Dict]) -> Dict[str, Dict]:
        """Insert or update multiple key-value pairs"""
        if not data:
            return {}

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany('''
                    INSERT INTO kv_store (namespace, key, value, updated_at)
                    VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                    ON CONFLICT (namespace, key)
                    DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = CURRENT_TIMESTAMP;
                ''', [(self.namespace, k, json.dumps(v)) for k, v in data.items()])
        return data

    async def drop(self):
        """Delete all data for current namespace"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute('''
                DELETE FROM kv_store WHERE namespace = $1;
            ''', self.namespace)

    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None

@dataclass
class PostgresVectorDBStorage(BaseVectorStorage):
    """PostgreSQL-based vector storage implementation using pgvector"""

    pool: Optional[asyncpg.Pool] = None
    _pool_lock = None

    def __post_init__(self):
        """Initialize connection pool and create table if needed"""
        import asyncio
        self._pool_lock = asyncio.Lock()
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool with retry logic"""
        if self.pool is None:
            async with self._pool_lock:
                if self.pool is None:
                    try:
                        self.pool = await asyncpg.create_pool(**self.global_config["postgres"])
                        await self._init_table()
                    except Exception as e:
                        logger.error(f"Failed to initialize PostgreSQL connection: {e}")
                        raise
        return self.pool

    async def _init_table(self):
        """Create the vector store table with pgvector extension"""
        async with self.pool.acquire() as conn:
            # Enable vector extension
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')

            # Create vector store table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS vector_store (
                    namespace TEXT,
                    id TEXT,
                    content TEXT,
                    embedding vector(384),
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (namespace, id)
                );

                CREATE INDEX IF NOT EXISTS idx_vector_store_namespace_id
                ON vector_store(namespace, id);

                CREATE INDEX IF NOT EXISTS idx_vector_store_embedding
                ON vector_store USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            ''')
            logger.info(f"Initialized PostgreSQL vector store table for namespace '{self.namespace}'")

    async def query(self, query: str, top_k: int) -> list[dict]:
        """Query vectors by similarity"""
        pool = await self._get_pool()

        # Get query embedding
        embeddings = await self.embedding_func([query])
        # Convert numpy array to vector string format
        query_vector = f"[{','.join(str(x) for x in embeddings[0].tolist())}]"

        async with pool.acquire() as conn:
            # Query similar vectors using cosine similarity
            rows = await conn.fetch('''
                SELECT id, content, metadata,
                       1 - (embedding <=> $3::vector) as similarity
                FROM vector_store
                WHERE namespace = $1
                ORDER BY embedding <=> $3::vector
                LIMIT $2;
            ''', self.namespace, top_k, query_vector)

            results = []
            for row in rows:
                result = {
                    'id': row['id'],
                    'content': row['content'],
                    'distance': 1 - row['similarity']  # Convert similarity to distance
                }
                if row['metadata']:
                    result.update(row['metadata'])
                results.append(result)

            return results

    async def upsert(self, data: dict[str, dict]):
        """Insert or update vectors"""
        if not data:
            return

        pool = await self._get_pool()

        # Get embeddings for all content
        contents = [item['content'] for item in data.values()]
        embeddings = await self.embedding_func(contents)

        async with pool.acquire() as conn:
            async with conn.transaction():
                # Prepare data for batch insert
                rows = []
                for i, (id, item) in enumerate(data.items()):
                    metadata = {k: v for k, v in item.items() if k != 'content'}
                    if self.meta_fields:
                        metadata = {k: v for k, v in metadata.items() if k in self.meta_fields}

                    # Convert numpy array to string format that pgvector expects
                    vector_str = f"[{','.join(str(x) for x in embeddings[i].tolist())}]"

                    rows.append((
                        self.namespace,
                        id,
                        item['content'],
                        vector_str,
                        json.dumps(metadata) if metadata else None
                    ))

                # Batch upsert
                await conn.executemany('''
                    INSERT INTO vector_store (namespace, id, content, embedding, metadata)
                    VALUES ($1, $2, $3, $4::vector, $5)
                    ON CONFLICT (namespace, id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata;
                ''', rows)

    async def drop(self):
        """Delete all vectors for current namespace"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute('''
                DELETE FROM vector_store WHERE namespace = $1;
            ''', self.namespace)

    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None

@dataclass
class PostgresGraphStorage(BaseGraphStorage):
    """PostgreSQL-based graph storage implementation"""

    def __post_init__(self):
        self._pool = None

    async def has_node(self, node_id: str) -> bool:
        # Implement node existence check
        return False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        # Implement edge existence check
        return False

    async def node_degree(self, node_id: str) -> int:
        # Implement node degree calculation
        return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        # Implement edge degree calculation
        return 0

    async def get_node(self, node_id: str) -> Union[dict, None]:
        # Implement node retrieval
        return None

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        # Implement edge retrieval
        return None

    async def get_node_edges(self, source_node_id: str) -> Union[list[tuple[str, str]], None]:
        # Implement node edges retrieval
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        # Implement node insertion/update
        pass

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]):
        # Implement edge insertion/update
        pass

    async def delete_node(self, node_id: str):
        # Implement node deletion
        pass

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        # Implement node embedding
        raise NotImplementedError("Node embedding is not implemented for PostgreSQL storage.")
