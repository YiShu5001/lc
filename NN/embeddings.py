"""Compatibility bridge for the refactored planning embedding modules."""

from src.lc.planning.models.embeddings import NeighborEmbedding, ObstacleEmbedding, SelfEmbedding

__all__ = ["NeighborEmbedding", "ObstacleEmbedding", "SelfEmbedding"]
