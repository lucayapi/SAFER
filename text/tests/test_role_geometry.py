"""Exact pair accounting for the existing-embedding role diagnostic."""

import numpy as np
import pandas as pd

from scripts.diagnose_role_geometry import cross_similarity, summarize_vectors, within_similarity


def test_exact_cosine_means_exclude_same_accident():
    roles = ["A0", "A1", "B", "C"]
    reference = pd.DataFrame({
        "accident_id": ["x", "x", "y", "z", "u", "v"],
        "pred_label": ["A0", "A0", "A0", "A0", *roles[1:3]],
    })
    vectors = np.array([[1, 0], [1, 0], [0, 1], [1, 1],
                        [0, 1], [1, 0]], dtype=float)
    # Add one unit of the missing fourth role for the summary precondition.
    reference.loc[len(reference)] = ["w", "C"]
    vectors = np.vstack([vectors, [1, 0]])
    stats = summarize_vectors(vectors, reference)
    unit = vectors[:4] / np.linalg.norm(vectors[:4], axis=1)[:, None]
    accidents = reference.accident_id[:4].to_numpy()
    brute = [float(np.dot(unit[i], unit[j])) for i in range(4) for j in range(i + 1, 4)
             if accidents[i] != accidents[j]]
    assert np.isclose(within_similarity(stats["A0"]), np.mean(brute))


def test_cross_cosine_mean():
    left_meta = pd.DataFrame({"accident_id": ["a", "b", "c", "d"],
                              "pred_label": ["A0", "A1", "B", "C"]})
    right_meta = pd.DataFrame({"accident_id": ["e", "f", "g", "h"],
                               "pred_label": ["A0", "A1", "B", "C"]})
    left = summarize_vectors(np.array([[1, 0], [0, 1], [1, 0], [0, 1]]), left_meta)
    right = summarize_vectors(np.array([[1, 1], [0, 1], [1, 0], [0, 1]]), right_meta)
    assert np.isclose(cross_similarity(left["A0"], right["A0"]), 1 / np.sqrt(2))
