import asyncio
import html
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
from typing import Any, Union, cast
import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB

from .utils import (
    logger,
    load_json,
    write_json,
    compute_mdhash_id,
)

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    # Default embedding dimension if not specified
    DEFAULT_EMBEDDING_DIM: int = 1536

    def __post_init__(self):
        self.embedding_dim = (
            getattr(self.embedding_func, "embedding_dim", None) or
            self.global_config.get("embedding_dim", self.DEFAULT_EMBEDDING_DIM)
        )

        if not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
            raise ValueError(f"Invalid embedding dimension: {self.embedding_dim}")

        self.working_dir = self.global_config["working_dir"]
        self._data_cache = {}  # Cache per user_id

    def _get_user_file_path(self, user_id: str = "default") -> str:
        """Get the file path for a specific user's data"""
        user_dir = os.path.join(self.working_dir, user_id)
        return os.path.join(user_dir, f"kv_store_{self.namespace}.json")

    def _ensure_user_directory(self, user_id: str = "default"):
        """Ensure the user's directory exists"""
        user_dir = os.path.dirname(self._get_user_file_path(user_id))
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            logger.info(f"Created user directory: {user_dir}")

    def _load_user_data(self, user_id: str = "default") -> dict:
        """Load data for a specific user"""
        if user_id not in self._data_cache:
            file_path = self._get_user_file_path(user_id)
            if os.path.exists(file_path):
                self._data_cache[user_id] = load_json(file_path) or {}
            else:
                self._data_cache[user_id] = {}
        return self._data_cache[user_id]

    async def all_keys(self, user_id: str = "default") -> list[str]:
        return list(self._load_user_data(user_id).keys())

    async def get_by_id(self, id: str, user_id: str = "default"):
        return self._load_user_data(user_id).get(id)

    async def get_by_ids(self, ids: list[str], fields=None, user_id: str = "default"):
        user_data = self._load_user_data(user_id)
        if fields is None:
            return [user_data.get(id) for id in ids]
        return [
            {k: v for k, v in user_data[id].items() if k in fields}
            if id in user_data else None
            for id in ids
        ]

    async def filter_keys(self, data: list[str], user_id: str = "default") -> set[str]:
        user_data = self._load_user_data(user_id)
        return set(k for k in data if k not in user_data)

    async def upsert(self, data: dict[str, dict], user_id: str = "default"):
        user_data = self._load_user_data(user_id)
        user_data.update(data)
        self._ensure_user_directory(user_id)
        write_json(user_data, self._get_user_file_path(user_id))
        return data

    async def drop(self, user_id: str = "default"):
        """Remove all data for a specific user"""
        if user_id in self._data_cache:
            del self._data_cache[user_id]

        file_path = self._get_user_file_path(user_id)
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove user directory if empty
        user_dir = os.path.dirname(file_path)
        if os.path.exists(user_dir) and not os.listdir(user_dir):
            os.rmdir(user_dir)


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2
    embedding_dim: int = 384

    def __post_init__(self):
        self.working_dir = self.global_config["working_dir"]
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._clients = {}
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )
        if hasattr(self.embedding_func, 'embedding_dim'):
            self.embedding_dim = self.embedding_func.embedding_dim

        # Initialize default user's storage
        self._get_client("default")

    def _get_client(self, user_id: str = "default") -> NanoVectorDB:
        if user_id not in self._clients:
            user_dir = os.path.join(self.working_dir, user_id)
            os.makedirs(user_dir, exist_ok=True)
            storage_file = os.path.join(user_dir, f"vdb_{self.namespace}.json")
            self._clients[user_id] = NanoVectorDB(
                self.embedding_dim,
                storage_file=storage_file
            )
        return self._clients[user_id]

    async def upsert(self, data: dict[str, dict], user_id: str = "default"):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []

        list_data = []
        for k, v in data.items():
            metadata = {k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields}
            metadata.update({
                "content": v["content"],
                "original_id": k  # Store original ID
            })
            list_data.append({
                "__id__": k,  # Use original ID as __id__
                "user_id": user_id,
                **metadata
            })

        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) == len(list_data):
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            client = self._get_client(user_id)
            client.upsert(datas=list_data)
            return list_data
        else:
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(self, query: str, top_k=5, user_id: str = "default"):
        client = self._get_client(user_id)
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        return [
            {
                **dp,
                "id": dp.get("original_id", dp["__id__"]),  # Use original ID if available
                "distance": dp["__metrics__"]
            }
            for dp in results
        ]

    async def index_done_callback(self):
        for client in self._clients.values():
            client.save()

    async def drop(self, user_id: str = "default"):
        if user_id in self._clients:
            client_file = self._clients[user_id].storage_file  # Use storage_file instead of _storage_file
            del self._clients[user_id]
            if os.path.exists(client_file):
                os.remove(client_file)

    @property
    def client_storage(self):
        return getattr(self._get_client("default"), "_NanoVectorDB__storage")

    async def delete_entity(self, entity_name: str, user_id: str = "default"):
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            entity_data = self._client.get([entity_id])

            if entity_data and entity_data[0].get("user_id") == user_id:
                self._client.delete([entity_id])
                logger.info(f"Entity {entity_name} has been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name} for user {user_id}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_relation(self, entity_name: str, user_id: str = "default"):
        try:
            relations = [
                dp
                for dp in self.client_storage["data"]
                if (dp["src_id"] == entity_name or dp["tgt_id"] == entity_name) and
                dp.get("user_id", "default") == user_id
            ]
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                self._client.delete(ids_to_delete)
                logger.info(
                    f"All relations related to entity {entity_name} for user {user_id} have been deleted."
                )
            else:
                logger.info(f"No relations found for entity {entity_name} and user {user_id}.")
        except Exception as e:
            logger.error(
                f"Error while deleting relations for entity {entity_name}: {e}"
            )


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    def _filter_by_user(self, data: dict, user_id: str) -> bool:
        """Helper method to check if a node/edge belongs to a user"""
        return data.get("user_id", "default") == user_id

    async def has_node(self, node_id: str, user_id: str = "default") -> bool:
        if not self._graph.has_node(node_id):
            return False
        return self._filter_by_user(self._graph.nodes[node_id], user_id)

    async def has_edge(self, source_node_id: str, target_node_id: str, user_id: str = "default") -> bool:
        if not self._graph.has_edge(source_node_id, target_node_id):
            return False
        return self._filter_by_user(self._graph.edges[source_node_id, target_node_id], user_id)

    async def get_node(self, node_id: str, user_id: str = "default") -> Union[dict, None]:
        node_data = self._graph.nodes.get(node_id)
        if node_data and self._filter_by_user(node_data, user_id):
            return node_data
        return None

    async def node_degree(self, node_id: str, user_id: str = "default") -> int:
        if not await self.has_node(node_id, user_id):
            return 0
        return sum(
            1 for _, _, data in self._graph.edges(node_id, data=True)
            if self._filter_by_user(data, user_id)
        )

    async def edge_degree(self, src_id: str, tgt_id: str, user_id: str = "default") -> int:
        src_degree = await self.node_degree(src_id, user_id)
        tgt_degree = await self.node_degree(tgt_id, user_id)
        return src_degree + tgt_degree

    async def get_edge(
        self, source_node_id: str, target_node_id: str, user_id: str = "default"
    ) -> Union[dict, None]:
        edge_data = self._graph.edges.get((source_node_id, target_node_id))
        if edge_data and self._filter_by_user(edge_data, user_id):
            return edge_data
        return None

    async def get_node_edges(
        self, source_node_id: str, user_id: str = "default"
    ) -> Union[list[tuple[str, str]], None]:
        if not await self.has_node(source_node_id, user_id):
            return None
        return [
            (source_node_id, target)
            for target in self._graph.neighbors(source_node_id)
            if self._filter_by_user(self._graph.edges[source_node_id, target], user_id)
        ]

    async def upsert_node(self, node_id: str, node_data: dict[str, str], user_id: str = "default"):
        node_data["user_id"] = user_id
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str], user_id: str = "default"
    ):
        edge_data["user_id"] = user_id
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str, user_id: str = "default"):
        if await self.has_node(node_id, user_id):
            self._graph.remove_node(node_id)
            logger.info(f"Node {node_id} deleted from the graph for user {user_id}.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for user {user_id}.")

    async def embed_nodes(self, algorithm: str, user_id: str = "default") -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")

        # Create a subgraph for the user's nodes
        user_nodes = [
            node for node, data in self._graph.nodes(data=True)
            if self._filter_by_user(data, user_id)
        ]
        user_subgraph = self._graph.subgraph(user_nodes)

        # Run embedding on the subgraph
        embeddings, nodes = await self._node_embed_algorithms[algorithm](user_subgraph)
        return embeddings, nodes

    async def _node2vec_embed(self, graph=None):
        from graspologic import embed

        if graph is None:
            graph = self._graph

        embeddings, nodes = embed.node2vec_embed(
            graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
