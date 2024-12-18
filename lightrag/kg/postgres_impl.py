import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Union, List, Dict, Any, Set
import psycopg
import psycopg_pool
import logging
from psycopg.rows import dict_row
import json

from ..base import (
    BaseKVStorage,
    #BaseVectorStorage,
)

logger = logging.getLogger(__name__)

###############################################################################
# Postgres Database Helper
###############################################################################

class PostgresDB:
    """
    A thin wrapper around a psycopg_pool.AsyncConnectionPool to execute queries.
    This class is similar in spirit to the OracleDB class provided, but for Postgres.

    It provides:
    - A connection pool
    - Methods to ensure required tables exist (check_tables)
    - Methods to drop tables (drop)
    - Simple query and execute methods
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PostgresDB with a configuration dictionary.
        """
        self.host = config.get("host")
        self.port = config.get("port", 5432)
        self.user = config.get("user")
        self.password = config.get("password")
        self.database = config.get("database")
        self.workspace = config.get("workspace", "default_workspace")

        if not all([self.host, self.user, self.password, self.database]):
            raise ValueError("Missing required database connection parameters.")

        # Build DSN
        self.dsn = (
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )

        # Create connection pool
        self._pool = psycopg_pool.AsyncConnectionPool(
            self.dsn,
            min_size=1,
            max_size=10,
            timeout=30,
            max_waiting=10
        )

    @property
    def pool(self):
        return self._pool

    async def close(self):
        await self._pool.close()


###############################################################################
# Schema Definitions
#
# These are analogous to the Oracle schema definitions, but for Postgres.
# We'll use a similar schema and naming.
#
# Note:
# - 'VECTOR' type in Oracle is replaced by 'VECTOR(1536)' in Postgres (example dimension)
#   You must adjust dimension according to your embedding function dimension.
# - We'll assume dimension from global_config["embedding_dim"] is available and used below.
###############################################################################

# We'll assume embeddings have a fixed dimension. For demonstration, let's say 1536.
# You should set this according to your actual embedding dimension.
DEFAULT_EMBEDDING_DIM = 1536

# Tables for KV and Vector store (similar to LIGHTRAG_DOC_FULL and LIGHTRAG_DOC_CHUNKS)
# We assume a structure similar to Oracle's.
LIST_OF_TABLES = [
    "lightrag_doc_full",
    "lightrag_doc_chunks",
    "lightrag_test",
]

# Constants for DDL
FULL_DOCS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id TEXT PRIMARY KEY,
    content TEXT,
    workspace TEXT,
    meta JSONB,
    createtime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updatetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

TEXT_CHUNKS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id TEXT PRIMARY KEY,
    content TEXT,
    tokens INT,
    chunk_order_index INT,
    full_doc_id TEXT,
    content_vector vector({embedding_dim}),
    workspace TEXT,
    meta JSONB,
    createtime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updatetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

BATCH_UPSERT_FULL_DOCS = """
INSERT INTO {table_name} (id, content, workspace, meta)
SELECT
    d.id,
    d.content,
    d.workspace,
    d.meta::jsonb
FROM jsonb_to_recordset(%s) AS d(
    id TEXT,
    content TEXT,
    workspace TEXT,
    meta JSONB
)
ON CONFLICT (id) DO UPDATE SET
    content = EXCLUDED.content,
    workspace = EXCLUDED.workspace,
    meta = EXCLUDED.meta,
    updatetime = CURRENT_TIMESTAMP;
"""

BATCH_UPSERT_CHUNKS = """
INSERT INTO {table_name} (
    id, content, tokens, chunk_order_index,
    full_doc_id, workspace, meta, content_vector
)
SELECT
    d.id,
    d.content,
    d.tokens,
    d.chunk_order_index,
    d.full_doc_id,
    d.workspace,
    d.meta::jsonb,
    d.content_vector::vector({embedding_dim})
FROM jsonb_to_recordset(%s) AS d(
    id TEXT,
    content TEXT,
    tokens INTEGER,
    chunk_order_index INTEGER,
    full_doc_id TEXT,
    workspace TEXT,
    meta JSONB,
    content_vector TEXT
)
ON CONFLICT (id) DO UPDATE SET
    content = EXCLUDED.content,
    tokens = EXCLUDED.tokens,
    chunk_order_index = EXCLUDED.chunk_order_index,
    full_doc_id = EXCLUDED.full_doc_id,
    workspace = EXCLUDED.workspace,
    meta = EXCLUDED.meta,
    content_vector = EXCLUDED.content_vector::vector({embedding_dim}),
    updatetime = CURRENT_TIMESTAMP;
"""

###############################################################################
# Helper functions
###############################################################################

def format_ids_for_in_clause(ids: List[str]) -> str:
    # Safely format an IN clause
    return ", ".join(["%s"] * len(ids))  # We'll pass ids as parameters separately

###############################################################################
# KV Storage Implementation for Postgres
###############################################################################

@dataclass
class PostgresKVStorage(BaseKVStorage):
    """
    Postgres-based KV Storage.

    - Uses two tables: lightrag_doc_full (for full_docs) and lightrag_doc_chunks (for text_chunks)
    - On upsert, embeddings are only generated for text_chunks (as in the Oracle example)
    """

    db: 'PostgresDB' = None  # Database connection
    embedding_dim: int = 1536  # Set your embedding dimension here

    # Default embedding dimension if not specified
    DEFAULT_EMBEDDING_DIM: int = 1536

    def __post_init__(self):
        # Validate embedding dimension first
        self.embedding_dim = (
            getattr(self.embedding_func, "embedding_dim", None) or
            self.global_config.get("embedding_dim", self.DEFAULT_EMBEDDING_DIM)
        )

        if not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
            raise ValueError(f"Invalid embedding dimension: {self.embedding_dim}")

        # Then proceed with other initialization
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)
        self.db = PostgresDB(self.global_config)
        self.namespace_table_map = {
            "test": "lightrag_test",
            "full_docs": "lightrag_doc_full",
            "text_chunks": "lightrag_doc_chunks",
        }

    async def check_tables(self):
        """Ensure that the necessary tables exist in the database."""
        table_name = self.namespace_table_map.get(self.namespace)
        if not table_name:
            return

        async with self.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                if self.namespace == "full_docs" or self.namespace == "test":
                    await cur.execute(FULL_DOCS_TABLE_DDL.format(table_name=table_name))
                elif self.namespace == "text_chunks":
                    await cur.execute(TEXT_CHUNKS_TABLE_DDL.format(
                        table_name=table_name,
                        embedding_dim=self.embedding_dim
                    ))
                await conn.commit()

    async def drop(self):
        """Drop all tables."""
        table_name = self.namespace_table_map.get(self.namespace)
        if not table_name:
            return

        async with self.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                DROP TABLE IF EXISTS {table_name}
                """)
                await conn.commit()

    async def get_by_id(self, id: str) -> Union[dict, None]:
        """Get a document by its ID."""
        table_name = self.namespace_table_map.get(self.namespace)
        if not table_name:
            return None

        async with self.db.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(f"""
                SELECT * FROM {table_name} WHERE id = %s
                """, (id,))
                row = await cur.fetchone()
                if not row:
                    return None

                # Reconstruct document from row
                result = {}
                if row.get('meta'):
                    result.update(row['meta'])
                if row.get('content'):
                    result['content'] = row['content']
                if row.get('workspace'):
                    result['workspace'] = row['workspace']
                if row.get('tokens') is not None:
                    result['tokens'] = row['tokens']
                if row.get('chunk_order_index') is not None:
                    result['chunk_order_index'] = row['chunk_order_index']
                if row.get('full_doc_id'):
                    result['full_doc_id'] = row['full_doc_id']
                return result

    async def upsert(self, docs: Dict[str, Dict]):
        """Insert or update documents."""
        table_name = self.namespace_table_map.get(self.namespace)
        if not table_name:
            return

        async with self.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                for doc_id, doc in docs.items():
                    # Extract special fields
                    content = doc.get('content', '')
                    workspace = doc.get('workspace', '')

                    # For text chunks, handle additional fields
                    tokens = doc.get('tokens')
                    chunk_order_index = doc.get('chunk_order_index')
                    full_doc_id = doc.get('full_doc_id')

                    # Store remaining fields in meta
                    meta = {k: v for k, v in doc.items()
                           if k not in ['content', 'workspace', 'tokens',
                                      'chunk_order_index', 'full_doc_id']}

                    if self.namespace == "text_chunks":
                        await cur.execute(f"""
                        INSERT INTO {table_name}
                        (id, content, tokens, chunk_order_index, full_doc_id,
                         workspace, meta, content_vector)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, NULL)
                        ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        tokens = EXCLUDED.tokens,
                        chunk_order_index = EXCLUDED.chunk_order_index,
                        full_doc_id = EXCLUDED.full_doc_id,
                        workspace = EXCLUDED.workspace,
                        meta = EXCLUDED.meta,
                        updatetime = CURRENT_TIMESTAMP
                        """, (doc_id, content, tokens, chunk_order_index,
                             full_doc_id, workspace, json.dumps(meta)))
                    else:
                        await cur.execute(f"""
                        INSERT INTO {table_name}
                        (id, content, workspace, meta)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        workspace = EXCLUDED.workspace,
                        meta = EXCLUDED.meta,
                        updatetime = CURRENT_TIMESTAMP
                        """, (doc_id, content, workspace, json.dumps(meta)))
                await conn.commit()

    async def get_by_ids(self, ids: list[str], fields: Union[set[str], None] = None) -> list[Union[dict, None]]:
        """Get multiple documents by their IDs."""
        table_name = self.namespace_table_map.get(self.namespace)
        if not table_name:
            return [None] * len(ids)

        async with self.db.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(f"""
                SELECT * FROM {table_name} WHERE id = ANY(%s)
                """, (ids,))
                rows = await cur.fetchall()

                # Create a mapping of id to row
                row_map = {row['id']: row for row in rows}

                # Return results in same order as input ids
                results = []
                for id in ids:
                    row = row_map.get(id)
                    if not row:
                        results.append(None)
                        continue

                    # Start with meta fields
                    result = row.get('meta', {})

                    # Only include requested fields
                    if fields is not None:
                        result = {k: v for k, v in result.items() if k in fields}

                    results.append(result)
                return results

    async def all_keys(self) -> list[str]:
        """Get all document IDs."""
        table_name = self.namespace_table_map.get(self.namespace)
        if not table_name:
            return []

        async with self.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                SELECT id FROM {table_name}
                """)
                rows = await cur.fetchall()
                return [row[0] for row in rows]

    async def filter_keys(self, data: list[str]) -> set[str]:
        """Return keys that don't exist in the database."""
        table_name = self.namespace_table_map.get(self.namespace)
        if not table_name:
            return set(data)

        async with self.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                SELECT id FROM {table_name}
                WHERE id = ANY(%s)
                """, (data,))

                existing_keys = {row[0] for row in await cur.fetchall()}
                return set(data) - existing_keys

    async def batch_upsert_nodes(self, nodes: Dict[str, Dict]):
        """
        Batch upsert nodes into storage.

        Args:
            nodes: Dictionary mapping node IDs to their data
        """
        if not nodes:
            return

        table_name = self.namespace_table_map.get(self.namespace)
        if not table_name:
            return

        batch_data = []

        # Process embeddings if needed for text chunks
        if self.namespace == "text_chunks":
            contents = [doc.get('content', '') for doc in nodes.values()]
            batches = [
                contents[i:i + self._max_batch_size]
                for i in range(0, len(contents), self._max_batch_size)
            ]

            # Generate embeddings in parallel
            embeddings_list = await asyncio.gather(
                *[self.embedding_func(batch) for batch in batches]
            )
            embeddings = np.concatenate(embeddings_list)

            # Prepare batch data with embeddings
            for i, (node_id, node) in enumerate(nodes.items()):
                batch_data.append({
                    'id': node_id,
                    'content': node.get('content', ''),
                    'tokens': node.get('tokens'),
                    'chunk_order_index': node.get('chunk_order_index'),
                    'full_doc_id': node.get('full_doc_id'),
                    'workspace': node.get('workspace', self.db.workspace),
                    'meta': {k: v for k, v in node.items() if k not in [
                        'content', 'workspace', 'tokens',
                        'chunk_order_index', 'full_doc_id'
                    ]},
                    'content_vector': embeddings[i].tolist()
                })

            # Execute batch upsert
            async with self.db.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        BATCH_UPSERT_CHUNKS.format(
                            table_name=table_name,
                            embedding_dim=self.embedding_dim
                        ),
                        [json.dumps(batch_data)]
                    )
                    await conn.commit()

        else:  # full_docs
            # Prepare batch data
            for node_id, node in nodes.items():
                batch_data.append({
                    'id': node_id,
                    'content': node.get('content', ''),
                    'workspace': node.get('workspace', self.db.workspace),
                    'meta': {k: v for k, v in node.items() if k not in ['content', 'workspace']}
                })

            # Execute batch upsert
            async with self.db.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        BATCH_UPSERT_FULL_DOCS.format(table_name=table_name),
                        [json.dumps(batch_data)]
                    )
                    await conn.commit()

    async def batch_get(self, ids: List[str], fields: Set[str] = None) -> Dict[str, Dict]:
        """
        Batch get documents by their IDs.

        Args:
            ids: List of document IDs to retrieve
            fields: Optional set of fields to include in results

        Returns:
            Dictionary mapping document IDs to their data, with None for missing IDs
        """
        if not ids:
            return {}

        table_name = self.namespace_table_map.get(self.namespace)
        if not table_name:
            return {id: None for id in ids}

        async with self.db.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(f"""
                SELECT * FROM {table_name}
                WHERE id = ANY(%s)
                """, (ids,))

                rows = await cur.fetchall()

                # Create result dictionary with None for missing keys
                results = {id: None for id in ids}

                # Update with found records
                for row in rows:
                    doc_id = row['id']

                    # Start with meta fields
                    result = row.get('meta', {}) or {}

                    # Add standard fields
                    if row.get('content'):
                        result['content'] = row['content']
                    if row.get('workspace'):
                        result['workspace'] = row['workspace']

                    # Add text_chunks specific fields
                    if self.namespace == "text_chunks":
                        if row.get('tokens') is not None:
                            result['tokens'] = row['tokens']
                        if row.get('chunk_order_index') is not None:
                            result['chunk_order_index'] = row['chunk_order_index']
                        if row.get('full_doc_id'):
                            result['full_doc_id'] = row['full_doc_id']

                    # Filter fields if specified
                    if fields:
                        result = {k: v for k, v in result.items() if k in fields}

                    results[doc_id] = result

                return results

    # Implement other methods as needed...

    async def close(self):
        await self.db.close()
