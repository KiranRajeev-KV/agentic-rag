from agentic_rag.storage.qdrant import VECTOR_SIZE, ensure_child_chunk_collection


def test_qdrant_default_vector_size_is_openai_dim() -> None:
    assert VECTOR_SIZE == 1536


def test_ensure_collection_uses_requested_vector_size() -> None:
    class _Collections:
        collections = []

    class _Client:
        def __init__(self) -> None:
            self.created_size = None

        def get_collections(self):  # noqa: ANN001
            return _Collections()

        def create_collection(self, collection_name, vectors_config):  # noqa: ANN001
            del collection_name
            self.created_size = vectors_config.size

        def create_payload_index(self, collection_name, field_name, field_schema, wait):  # noqa: ANN001
            del collection_name, field_name, field_schema, wait

    client = _Client()
    ensure_child_chunk_collection(client=client, collection_name="c", vector_size=1536)
    assert client.created_size == 1536
