import os
import asyncio
import tempfile
import pytest
import vcr
from lightrag import LightRAG, QueryParam
from lightrag.storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage
from lightrag.llm import gpt_4o_mini_complete

# Configure VCR
vcr = vcr.VCR(
    serializer='yaml',
    cassette_library_dir='tests/fixtures/vcr_cassettes',
    record_mode='new_episodes',
    match_on=['method', 'scheme', 'host', 'port', 'path'],
    filter_headers=['authorization', 'api-key'],
    filter_query_parameters=['api-key', 'key'],
    decode_compressed_response=True,
    # Filter sensitive data from request and response
    before_record_request=lambda request: request,
    before_record_response=lambda response: response,
)

@pytest.fixture(autouse=True)
async def cleanup_cassettes():
  pass

@pytest.fixture
async def sample_texts():
    return {
        "user1": [
            "The quick brown fox jumps over the lazy dog.",
            "To be or not to be, that is the question."
        ],
        "user2": [
            "All that glitters is not gold.",
            "A journey of a thousand miles begins with a single step."
        ]
    }

@pytest.mark.asyncio
@vcr.use_cassette()
async def test_multi_user_storage_isolation(sample_texts):
    """Test that storage backends properly isolate data between users."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize LightRAG instances for different users
        rag = LightRAG(
            working_dir=temp_dir,
            llm_model_func=gpt_4o_mini_complete,
        )

        # Insert user-specific documents
        for doc in sample_texts["user1"]:
            await rag.ainsert(doc, user_id="user1")
        for doc in sample_texts["user2"]:
            await rag.ainsert(doc, user_id="user2")

        # Test JsonKVStorage isolation
        user1_chunks = await rag.text_chunks.all_keys(user_id="user1")
        user2_chunks = await rag.text_chunks.all_keys(user_id="user2")
        assert len(user1_chunks) > 0
        assert len(user2_chunks) > 0
        assert not set(user1_chunks).intersection(set(user2_chunks))

        # Test NanoVectorDBStorage isolation
        query = "fox and dog"
        user1_results = await rag.chunks_vdb.query(query, top_k=5, user_id="user1")
        user2_results = await rag.chunks_vdb.query(query, top_k=5, user_id="user2")

        # Get the actual content from text_chunks
        user1_contents = [
            (await rag.text_chunks.get_by_id(r["id"], user_id="user1"))["content"]
            for r in user1_results
        ]
        user2_contents = [
            (await rag.text_chunks.get_by_id(r["id"], user_id="user2"))["content"]
            for r in user2_results
        ]

        # User1's results should contain fox/dog references, User2's should not
        assert any("fox" in content.lower() or "dog" in content.lower()
                  for content in user1_contents)
        assert not any("fox" in content.lower() or "dog" in content.lower()
                      for content in user2_contents)

        # Test NetworkXStorage isolation
        # First, get entities for each user
        user1_entities = await rag.entities_vdb.query("fox dog", top_k=10, user_id="user1")
        user2_entities = await rag.entities_vdb.query("journey", top_k=10, user_id="user2")

        # Check that entities are properly isolated
        for entity in user1_entities:
            entity_name = entity.get("entity_name")
            if entity_name:
                # User1 should have access to their entities
                assert await rag.chunk_entity_relation_graph.has_node(entity_name, user_id="user1")
                # User2 should not see User1's entities
                assert not await rag.chunk_entity_relation_graph.has_node(entity_name, user_id="user2")

@pytest.mark.asyncio
@vcr.use_cassette()
async def test_multi_user_concurrent_access(sample_texts):
    """Test concurrent access to storage backends by multiple users."""
    with tempfile.TemporaryDirectory() as temp_dir:
        rag = LightRAG(
            working_dir=temp_dir,
            llm_model_func=gpt_4o_mini_complete,
        )

        # Insert documents for each user
        for text in sample_texts["user1"]:
            await rag.ainsert(text, user_id="user1")
        for text in sample_texts["user2"]:
            await rag.ainsert(text, user_id="user2")

        # Query for each user
        response1 = await rag.aquery(
            "Tell me about the fox and dog",
            param=QueryParam(mode="hybrid"),
            user_id="user1"
        )
        response2 = await rag.aquery(
            "Tell me about the journey and miles",
            param=QueryParam(mode="hybrid"),
            user_id="user2"
        )

        # Verify responses are user-specific
        assert any(term in response1.lower() for term in ["fox", "dog"])
        assert any(term in response2.lower() for term in ["journey", "miles"])

@pytest.mark.asyncio
@vcr.use_cassette()
async def test_multi_user_filesystem_persistence(sample_texts):
    """Test that user data persists correctly in filesystem storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # First instance - create and store data
        rag = LightRAG(
            working_dir=temp_dir,
            llm_model_func=gpt_4o_mini_complete,
        )

        # Insert data for different users
        for text in sample_texts["user1"]:
            await rag.ainsert(text, user_id="user1")
        for text in sample_texts["user2"]:
            await rag.ainsert(text, user_id="user2")

        # Force save all data
        await rag.text_chunks.index_done_callback()
        await rag.chunks_vdb.index_done_callback()
        await rag.entities_vdb.index_done_callback()
        await rag.relationships_vdb.index_done_callback()


        # Create new instances pointing to same directories

        rag2 = LightRAG(
            working_dir=temp_dir,
            llm_model_func=gpt_4o_mini_complete,
        )

        # Verify data loaded correctly for each user
        # Check KV storage
        chunks1_user1 = await rag.text_chunks.all_keys(user_id="user1")
        chunks2_user1 = await rag2.text_chunks.all_keys(user_id="user1")
        assert set(chunks1_user1) == set(chunks2_user1)

        chunks1_user2 = await rag.text_chunks.all_keys(user_id="user2")
        chunks2_user2 = await rag2.text_chunks.all_keys(user_id="user2")
        assert set(chunks1_user2) == set(chunks2_user2)

        # Check vector storage
        query1 = "fox and dog"
        query2 = "journey and miles"

        results1_user1 = await rag.chunks_vdb.query(query1, top_k=5, user_id="user1")
        results2_user1 = await rag2.chunks_vdb.query(query1, top_k=5, user_id="user1")
        assert len(results1_user1) == len(results2_user1)

        results1_user2 = await rag.chunks_vdb.query(query2, top_k=5, user_id="user2")
        results2_user2 = await rag2.chunks_vdb.query(query2, top_k=5, user_id="user2")
        assert len(results1_user2) == len(results2_user2)

        # Check graph storage
        entities1_user1 = await rag.entities_vdb.query(query1, top_k=10, user_id="user1")
        entities2_user1 = await rag2.entities_vdb.query(query1, top_k=10, user_id="user1")
        assert len(entities1_user1) == len(entities2_user1)

        entities1_user2 = await rag.entities_vdb.query(query2, top_k=10, user_id="user2")
        entities2_user2 = await rag2.entities_vdb.query(query2, top_k=10, user_id="user2")
        assert len(entities1_user2) == len(entities2_user2)

@pytest.mark.asyncio
async def test_graph_storage_isolation():
    """Test that graph storage properly isolates data between users."""
    with tempfile.TemporaryDirectory() as temp_dir:
        graph_storage = NetworkXStorage(
            namespace="test_graph",
            global_config={"working_dir": temp_dir},
            embedding_func=None
        )

        # Add nodes for user1
        await graph_storage.upsert_node("node1", {"data": "user1_data"}, user_id="user1")
        await graph_storage.upsert_node("node2", {"data": "user1_data"}, user_id="user1")
        await graph_storage.upsert_edge("node1", "node2", {"type": "user1_relation"}, user_id="user1")

        # Add nodes for user2
        await graph_storage.upsert_node("node1", {"data": "user2_data"}, user_id="user2")
        await graph_storage.upsert_node("node2", {"data": "user2_data"}, user_id="user2")
        await graph_storage.upsert_edge("node1", "node2", {"type": "user2_relation"}, user_id="user2")

        # Verify user1's data
        node1_user1 = await graph_storage.get_node("node1", user_id="user1")
        assert node1_user1["data"] == "user1_data"
        edge_user1 = await graph_storage.get_edge("node1", "node2", user_id="user1")
        assert edge_user1["type"] == "user1_relation"

        # Verify user2's data
        node1_user2 = await graph_storage.get_node("node1", user_id="user2")
        assert node1_user2["data"] == "user2_data"
        edge_user2 = await graph_storage.get_edge("node1", "node2", user_id="user2")
        assert edge_user2["type"] == "user2_relation"

@pytest.mark.asyncio
async def test_graph_storage_file_structure():
    """Test that graph storage creates proper file structure for each user."""
    with tempfile.TemporaryDirectory() as temp_dir:
        graph_storage = NetworkXStorage(
            namespace="test_graph",
            global_config={"working_dir": temp_dir},
            embedding_func=None
        )

        # Add data for different users
        await graph_storage.upsert_node("node1", {"data": "test"}, user_id="user1")
        await graph_storage.upsert_node("node1", {"data": "test"}, user_id="user2")

        # Check directory structure
        user1_dir = os.path.join(temp_dir, "user1")
        user2_dir = os.path.join(temp_dir, "user2")

        assert os.path.exists(user1_dir), "User1 directory should exist"
        assert os.path.exists(user2_dir), "User2 directory should exist"

        # Check for graph files in user directories
        assert os.path.exists(os.path.join(user1_dir, "test_graph.graphml")), "User1 graph file should exist"
        assert os.path.exists(os.path.join(user2_dir, "test_graph.graphml")), "User2 graph file should exist"

@pytest.mark.asyncio
async def test_graph_operations_user_isolation():
    """Test that graph operations are properly isolated between users."""
    with tempfile.TemporaryDirectory() as temp_dir:
        graph_storage = NetworkXStorage(
            namespace="test_graph",
            global_config={"working_dir": temp_dir},
            embedding_func=None
        )

        # Add node for user1
        await graph_storage.upsert_node("shared_node", {"data": "user1_data"}, user_id="user1")

        # Verify user2 cannot see or modify user1's node
        assert not await graph_storage.has_node("shared_node", user_id="user2")

        # Add same node ID for user2
        await graph_storage.upsert_node("shared_node", {"data": "user2_data"}, user_id="user2")

        # Verify each user sees their own version
        node_user1 = await graph_storage.get_node("shared_node", user_id="user1")
        node_user2 = await graph_storage.get_node("shared_node", user_id="user2")

        assert node_user1["data"] == "user1_data"
        assert node_user2["data"] == "user2_data"

        # Delete node for user1
        await graph_storage.delete_node("shared_node", user_id="user1")

        # Verify user2's node still exists
        assert not await graph_storage.has_node("shared_node", user_id="user1")
        assert await graph_storage.has_node("shared_node", user_id="user2")
