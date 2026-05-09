from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models

VECTOR_SIZE = 1024
DISTANCE = models.Distance.COSINE

PAYLOAD_INDEX_FIELDS: dict[str, models.PayloadSchemaType] = {
    "paper_id": models.PayloadSchemaType.KEYWORD,
    "parent_id": models.PayloadSchemaType.KEYWORD,
    "arxiv_id": models.PayloadSchemaType.KEYWORD,
    "section_type": models.PayloadSchemaType.KEYWORD,
    "content_type": models.PayloadSchemaType.KEYWORD,
    "primary_category": models.PayloadSchemaType.KEYWORD,
    "published_at": models.PayloadSchemaType.DATETIME,
    "updated_at": models.PayloadSchemaType.DATETIME,
}


def get_client(url: str) -> QdrantClient:
    return QdrantClient(url=url)


def ensure_child_chunk_collection(client: QdrantClient, collection_name: str) -> None:
    collections = client.get_collections().collections
    existing = {collection.name for collection in collections}

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
        )

    for field_name, field_schema in PAYLOAD_INDEX_FIELDS.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )
