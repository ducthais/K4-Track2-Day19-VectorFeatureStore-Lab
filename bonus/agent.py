"""Hybrid Memory Agent combining Vector Store (Episodic) and Feast Feature Store (Profile).

Bonus challenge implementation for Day 19 Lab.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
FEAST_REPO = ROOT / "app" / "feast_repo"

class HybridMemoryAgent:
    """Agent blending long-term episodic memory (vector/hybrid search) with stable

    and streaming user features (Feast feature store).
    """

    def __init__(self, collection_name: str = "agent_memory"):
        self.collection_name = collection_name
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        # In-memory document storage for BM25 and content lookup
        self.memories: list[dict[str, Any]] = []
        self._point_id = 0

        # Try to connect to Feast online store
        self.feature_store = None
        try:
            from feast import FeatureStore
            if (FEAST_REPO / "registry.db").exists():
                self.feature_store = FeatureStore(repo_path=str(FEAST_REPO))
        except Exception:
            self.feature_store = None

    def remember(self, text: str, user_id: str = "u_001", metadata: dict | None = None) -> list[int]:
        """Store an episodic memory for a specific user.

        Chunks text if multi-line/long, embeds each chunk, and indexes in Qdrant + BM25.
        """
        # Chunking: split by paragraphs / logical sections
        raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        if not raw_chunks:
            raw_chunks = [text.strip()]

        created_ids = []
        for chunk in raw_chunks:
            self._point_id += 1
            pid = self._point_id
            vector = next(self.embedder.embed([chunk])).tolist()

            payload = {
                "id": pid,
                "user_id": user_id,
                "text": chunk,
                "timestamp": time.time(),
                **(metadata or {}),
            }
            self.memories.append(payload)

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=pid,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            created_ids.append(pid)

        return created_ids

    def _get_user_features(self, user_id: str) -> dict[str, Any]:
        """Fetch stable profile + streaming velocity features from Feast."""
        fallback = {
            "reading_speed_wpm": 220,
            "preferred_language": "vi",
            "topic_affinity": "cloud",
            "queries_last_hour": 5,
            "distinct_topics_24h": 3,
        }

        if not self.feature_store:
            return fallback

        features_to_fetch = [
            "user_profile_features:reading_speed_wpm",
            "user_profile_features:preferred_language",
            "user_profile_features:topic_affinity",
            "query_velocity_features:queries_last_hour",
            "query_velocity_features:distinct_topics_24h",
        ]

        try:
            res = self.feature_store.get_online_features(
                features=features_to_fetch,
                entity_rows=[{"user_id": user_id}],
            ).to_dict()

            profile = {}
            for k, v in res.items():
                short_k = k.split(":")[-1]
                val = v[0] if v and len(v) > 0 else None
                profile[short_k] = val if val is not None else fallback.get(short_k)
            return profile
        except Exception:
            return fallback

    def _search_user_memories(self, query: str, user_id: str, top_k: int = 3, rrf_k: int = 60) -> list[dict]:
        """Hybrid search (BM25 + Dense Vector + RRF) filtered strictly by user_id."""
        user_docs = [m for m in self.memories if m["user_id"] == user_id]
        if not user_docs:
            return []

        # 1. BM25 on user's episodic chunks
        tokenized_corpus = [doc["text"].lower().split() for doc in user_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query.lower().split())
        ranked_kw_indices = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])
        kw_ranked_ids = [user_docs[i]["id"] for i in ranked_kw_indices if bm25_scores[i] > 0]

        # 2. Dense Vector in Qdrant with payload filter for tenant isolation
        q_vec = next(self.embedder.embed([query])).tolist()
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            ]
        )
        vec_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            query_filter=user_filter,
            limit=max(top_k * 3, 10),
        ).points
        sem_ranked_ids = [h.id for h in vec_hits]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[int, float] = {}
        for rank, pid in enumerate(kw_ranked_ids, start=1):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rrf_k + rank)
        for rank, pid in enumerate(sem_ranked_ids, start=1):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rrf_k + rank)

        # Build top-k output
        sorted_pids = sorted(rrf_scores.keys(), key=lambda pid: -rrf_scores[pid])[:top_k]
        id_to_doc = {d["id"]: d for d in user_docs}
        results = []
        for pid in sorted_pids:
            doc = dict(id_to_doc[pid])
            doc["score"] = rrf_scores[pid]
            results.append(doc)
        return results

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Assemble complete context combining episodic memory and user profile."""
        profile = self._get_user_features(user_id)
        memories = self._search_user_memories(query, user_id=user_id, top_k=3)

        # Assemble prompt injection context
        lines = []
        lines.append("=== AI MEMORY CONTEXT ===")
        lines.append(f"User ID: {user_id}")
        lines.append(
            f"User Profile: preferred_lang={profile.get('preferred_language')}, "
            f"topic_affinity={profile.get('topic_affinity')}, "
            f"reading_speed={profile.get('reading_speed_wpm')} wpm"
        )
        lines.append(
            f"Recent Activity (Streaming): {profile.get('queries_last_hour', 0)} queries in last hour, "
            f"{profile.get('distinct_topics_24h', 1)} active topics in 24h"
        )
        lines.append("\nTop Relevant Episodic Memories:")
        if memories:
            for i, m in enumerate(memories, 1):
                preview = m['text'].replace('\n', ' ')
                lines.append(f"  [{i}] (RRF={m['score']:.4f}) {preview}")
        else:
            lines.append("  (No relevant past episodic memories found for this query)")

        lines.append("=========================")
        return "\n".join(lines)
