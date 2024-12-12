import asyncpg
import logging
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Union
import json

from ..base import BaseKVStorage
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
