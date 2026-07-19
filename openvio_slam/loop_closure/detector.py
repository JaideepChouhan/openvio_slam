"""
loop_closure/detector.py
========================
Loop closure candidate detection via BoW similarity.

Similarity threshold tuning:
  Too high (0.9+): missed loops → drift accumulates
  Too low  (0.3-): false positives → geometric verifier flooded
  Sweet spot: 0.5-0.7 for indoor single-room scenes
"""

import numpy as np
from typing import List, Tuple
from .vocabulary import ORBVocabulary


class LoopCandidateDatabase:
    """Stores keyframe BoW vectors and provides similarity search."""

    def __init__(self, vocabulary: ORBVocabulary,
                 similarity_threshold: float = 0.6,
                 min_temporal_gap: int = 20):
        self.vocab     = vocabulary
        self.threshold = similarity_threshold
        self.min_gap   = min_temporal_gap

        self._keyframe_ids:  List[int]        = []
        self._bow_vectors:   List[np.ndarray] = []
        self._idf_update_freq = 50

    def add_keyframe(self, keyframe_id: int, descriptors: np.ndarray) -> np.ndarray:
        bow_vec = self.vocab.compute_bow_vector(descriptors)
        self._keyframe_ids.append(keyframe_id)
        self._bow_vectors.append(bow_vec)
        if len(self._bow_vectors) % self._idf_update_freq == 0:
            self.vocab.update_idf(self._bow_vectors)
        return bow_vec

    def query(self, query_bow: np.ndarray, query_keyframe_id: int,
              top_k: int = 5) -> List[Tuple[int, float]]:
        if len(self._bow_vectors) < self.min_gap + 1:
            return []

        bow_mat = np.vstack(self._bow_vectors)
        sims    = bow_mat @ query_bow

        candidates = []
        for kf_id, sim in zip(self._keyframe_ids, sims):
            if abs(query_keyframe_id - kf_id) >= self.min_gap and sim >= self.threshold:
                candidates.append((kf_id, float(sim)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def __len__(self) -> int:
        return len(self._keyframe_ids)
