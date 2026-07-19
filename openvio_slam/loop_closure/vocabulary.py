"""
loop_closure/vocabulary.py
==========================
Bag-of-Words (BoW) vocabulary for place recognition.

How BoW works for SLAM:
    1. Offline: cluster a large set of ORB descriptors from diverse indoor images
       using hierarchical k-means. Each leaf node of the tree is a "visual word".
    2. Online: for each new keyframe, assign each ORB descriptor to its nearest
       visual word. The keyframe is a TF-IDF weighted histogram over all words.
    3. Loop detection: compare current keyframe BoW vector against all stored
       keyframe vectors using cosine similarity.

Our vocabulary: 2-level tree, k1=10, k2=10 → 100 visual words.
TF-IDF weighting downweights ubiquitous background words, upweights rare ones.

Reference: Galvez-Lopez & Tardos, "Bags of Binary Words for Fast Place Recognition
           in Image Sequences", IEEE TRO 2012.
"""

import numpy as np
import pickle
from typing import List, Optional


class BinaryKMeans:
    """K-means clustering for binary (ORB) descriptors using Hamming distance."""

    def __init__(self, k: int, max_iter: int = 20, n_restarts: int = 3):
        self.k          = k
        self.max_iter   = max_iter
        self.n_restarts = n_restarts
        self.centers_   = None

    def fit(self, descriptors: np.ndarray) -> 'BinaryKMeans':
        N = len(descriptors)
        if N <= self.k:
            self.centers_ = descriptors[:self.k].copy()
            if N < self.k:
                pad = np.random.randint(0, 256, (self.k - N, 32), dtype=np.uint8)
                self.centers_ = np.vstack([self.centers_, pad])
            return self

        best_inertia = float('inf')
        best_centers = None

        for _ in range(self.n_restarts):
            centers = self._init_kmeanspp(descriptors)
            prev_labels = None
            for _ in range(self.max_iter):
                labels = self._assign(descriptors, centers)
                centers = self._update_centers(descriptors, labels)
                if prev_labels is not None and np.array_equal(labels, prev_labels):
                    break
                prev_labels = labels
            inertia = self._compute_inertia(descriptors, labels, centers)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()

        self.centers_ = best_centers
        return self

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        return self._assign(descriptors, self.centers_)

    def _hamming_batch(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Pairwise Hamming distances: D[i,j] = Hamming(A[i], B[j])."""
        A_bits = np.unpackbits(A, axis=1).astype(np.int8)
        B_bits = np.unpackbits(B, axis=1).astype(np.int8)
        D = np.zeros((len(A), len(B)), dtype=np.int32)
        for i in range(len(A)):
            D[i] = np.sum(A_bits[i:i+1] != B_bits, axis=1)
        return D

    def _assign(self, descriptors, centers):
        D = self._hamming_batch(descriptors, centers)
        return np.argmin(D, axis=1)

    def _update_centers(self, descriptors, labels):
        centers = np.zeros((self.k, 32), dtype=np.uint8)
        for j in range(self.k):
            mask = labels == j
            if mask.sum() == 0:
                centers[j] = descriptors[np.random.randint(len(descriptors))]
            else:
                bits = np.unpackbits(descriptors[mask], axis=1)
                majority = (bits.sum(axis=0) > mask.sum() / 2).astype(np.uint8)
                centers[j] = np.packbits(majority)
        return centers

    def _init_kmeanspp(self, descriptors):
        N = len(descriptors)
        centers = [descriptors[np.random.randint(N)]]
        for _ in range(1, self.k):
            D = self._hamming_batch(descriptors, np.array(centers))
            min_dists = D.min(axis=1).astype(float)
            probs = min_dists / max(min_dists.sum(), 1e-10)
            centers.append(descriptors[np.random.choice(N, p=probs)])
        return np.array(centers, dtype=np.uint8)

    def _compute_inertia(self, descriptors, labels, centers):
        total = 0
        for i, label in enumerate(labels):
            bd = np.unpackbits(descriptors[i:i+1], axis=1)
            bc = np.unpackbits(centers[label:label+1], axis=1)
            total += int(np.sum(bd != bc))
        return total


class ORBVocabulary:
    """
    2-level hierarchical BoW vocabulary for ORB descriptors.
    k1×k2 visual words (default: 10×10 = 100).
    """

    def __init__(self, k1: int = 10, k2: int = 10):
        self.k1 = k1
        self.k2 = k2
        self.n_words = k1 * k2
        self._level0_kmeans: Optional[BinaryKMeans] = None
        self._level1_kmeans: List[BinaryKMeans] = []
        self._idf: Optional[np.ndarray] = None
        self._n_training_docs = 0
        self._is_trained = False

    def train(self, all_descriptors: np.ndarray, n_training_images: int = 1):
        print(f"Training vocabulary: {len(all_descriptors)} descriptors → {self.n_words} words...")
        self._level0_kmeans = BinaryKMeans(k=self.k1)
        self._level0_kmeans.fit(all_descriptors)
        labels_0 = self._level0_kmeans.predict(all_descriptors)

        self._level1_kmeans = []
        for i in range(self.k1):
            mask = labels_0 == i
            subset = all_descriptors[mask] if mask.sum() >= self.k2 else \
                     np.tile(all_descriptors, (self.k2 // max(1, len(all_descriptors)) + 2, 1))[:self.k2]
            km = BinaryKMeans(k=self.k2)
            km.fit(subset)
            self._level1_kmeans.append(km)

        self._idf = np.ones(self.n_words, dtype=float)
        self._n_training_docs = n_training_images
        self._is_trained = True
        print(f"Vocabulary trained: {self.k1}×{self.k2} = {self.n_words} words")

    def compute_bow_vector(self, descriptors: np.ndarray) -> np.ndarray:
        assert self._is_trained, "Vocabulary not trained."
        if len(descriptors) == 0:
            return np.zeros(self.n_words, dtype=float)
        word_ids = self._descriptors_to_word_ids(descriptors)
        tf = np.bincount(word_ids, minlength=self.n_words).astype(float)
        tf /= max(1.0, tf.sum())
        tfidf = tf * self._idf
        norm = np.linalg.norm(tfidf)
        if norm > 1e-8:
            tfidf /= norm
        return tfidf

    def update_idf(self, all_bow_vectors: List[np.ndarray]):
        if not all_bow_vectors:
            return
        N   = len(all_bow_vectors)
        mat = np.vstack(all_bow_vectors)
        df  = np.maximum((mat > 1e-8).sum(axis=0).astype(float), 1.0)
        self._idf = np.clip(np.log(N / df), 0.0, 10.0)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'k1': self.k1, 'k2': self.k2,
                         'level0': self._level0_kmeans,
                         'level1': self._level1_kmeans,
                         'idf': self._idf,
                         'n_docs': self._n_training_docs}, f)
        print(f"Vocabulary saved: {path}")

    @classmethod
    def load(cls, path: str) -> 'ORBVocabulary':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab = cls(k1=data['k1'], k2=data['k2'])
        vocab._level0_kmeans   = data['level0']
        vocab._level1_kmeans   = data['level1']
        vocab._idf             = data['idf']
        vocab._n_training_docs = data['n_docs']
        vocab._is_trained      = True
        print(f"Vocabulary loaded: {path} ({vocab.n_words} words)")
        return vocab

    def _descriptors_to_word_ids(self, descriptors: np.ndarray) -> np.ndarray:
        labels_0 = self._level0_kmeans.predict(descriptors)
        word_ids = np.zeros(len(descriptors), dtype=int)
        for i in range(self.k1):
            mask = labels_0 == i
            if mask.sum() == 0:
                continue
            labels_1 = self._level1_kmeans[i].predict(descriptors[mask])
            word_ids[mask] = i * self.k2 + labels_1
        return word_ids
