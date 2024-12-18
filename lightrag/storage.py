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
        # Validate embedding dimension first
        self.embedding_dim = (
            getattr(self.embedding_func, "embedding_dim", None) or
            self.global_config.get("embedding_dim", self.DEFAULT_EMBEDDING_DIM)
        )

        if not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
            raise ValueError(f"Invalid embedding dimension: {self.embedding_dim}")

        # Then proceed with existing initialization
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    def _strip_internal_fields(self, data: dict) -> dict:
        """Remove internal fields like user_id from the data for backward compatibility"""
        if data is None:
            return None
        return {k: v for k, v in data.items() if k != "user_id"}

    async def all_keys(self, user_id: str = "default") -> list[str]:
        return [k for k, v in self._data.items() if v.get("user_id", "default") == user_id]

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id: str, user_id: str = "default"):
        data = self._data.get(id)
        if data and data.get("user_id", "default") == user_id:
            return self._strip_internal_fields(data)
        return None

    async def get_by_ids(self, ids: list[str], fields=None, user_id: str = "default"):
        if fields is None:
            return [
                self._strip_internal_fields(self._data.get(id))
                if self._data.get(id) and self._data[id].get("user_id", "default") == user_id
                else None
                for id in ids
            ]
        return [
            (
                self._strip_internal_fields({k: v for k, v in self._data[id].items() if k in fields})
                if self._data.get(id) and self._data[id].get("user_id", "default") == user_id
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str], user_id: str = "default") -> set[str]:
        return set([
            s for s in data
            if s not in self._data or
            self._data[s].get("user_id", "default") != user_id
        ])

    async def upsert(self, data: dict[str, dict], user_id: str = "default"):
        left_data = {
            k: {**v, "user_id": user_id}
            for k, v in data.items()
            if k not in self._data or self._data[k].get("user_id", "default") != user_id
        }
        self._data.update(left_data)
        return {k: self._strip_internal_fields(v) for k, v in left_data.items()}

    async def drop(self, user_id: str = "default"):
        self._data = {
            k: v
            for k, v in self._data.items()
            if v.get("user_id", "default") != user_id
        }


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict], user_id: str = "default"):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                "user_id": user_id,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
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
            results = self._client.upsert(datas=list_data)
            return results
        else:
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(self, query: str, top_k=5, user_id: str = "default"):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        all_results = self._client.query(
            query=embedding,
            top_k=top_k * 2,  # Get more results since we'll filter by user_id
            better_than_threshold=self.cosine_better_than_threshold,
        )
        # Filter results by user_id and format them
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]}
            for dp in all_results
            if dp.get("user_id", "default") == user_id
        ][:top_k]  # Limit to original top_k after filtering
        return results

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

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

    async def index_done_callback(self):
        self._client.save()


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
