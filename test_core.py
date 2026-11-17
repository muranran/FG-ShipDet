"""CPU checks for source exclusion, graph propagation, calibration, and label barriers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

import fgship_core as core
from evaluate import evaluate


class CoreTests(unittest.TestCase):
    def setUp(self):
        core.DEVICE = "cpu"
        core.K = 3
        core.ALPHA = 0.9
        torch.set_num_threads(2)

    def features(self, n):
        rng = np.random.default_rng(18)
        g = rng.normal(size=(n, 8)).astype("float32")
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        l = rng.normal(size=(n, 3, 4)).astype("float32")
        l /= np.linalg.norm(l, axis=2, keepdims=True)
        return dict(
            global_features=g,
            local_features=l,
            paths=np.array([f"p{i}" for i in range(n)]),
            groups=np.array([f"g{i}" for i in range(n)]),
            pixel_hash=np.array([f"h{i}" for i in range(n)]),
        )

    def test_exclusion_and_row_normalization(self):
        f = self.features(20)
        f["groups"][1] = f["groups"][0]
        f["pixel_hash"][2] = f["pixel_hash"][0]
        p = core.graph(f, f)
        self.assertEqual(float(p[0, 0] + p[0, 1] + p[0, 2]), 0.0)
        np.testing.assert_allclose(np.asarray(p.sum(1)), 1)
        self.assertTrue(np.all(p.data > 0))

    def test_diffusion_against_dense_solution(self):
        p = sp.csr_matrix([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]], dtype=float)
        y = np.ones((1, 1))
        h, _ = core.diffuse(p, 1, y)
        expected = np.linalg.solve(
            np.eye(2) - 0.9 * p[1:, 1:].toarray(), 0.9 * p[1:, :1].toarray() @ y
        )
        np.testing.assert_allclose(h[1:], expected, atol=1e-8)

    def test_relative_distance_against_scalar(self):
        f = self.features(20)
        actual, _ = core.distances(f, f)
        for i in range(20):
            dg = np.maximum(1 - f["global_features"][i] @ f["global_features"].T, 0)
            dl = np.minimum(
                1
                - np.einsum("pc,npc->n", f["local_features"][i], f["local_features"])
                / 3,
                1
                - np.einsum(
                    "pc,npc->n", f["local_features"][i, ::-1], f["local_features"]
                )
                / 3,
            )
            dg[i] = np.inf
            dl[i] = np.inf
            self.assertAlmostEqual(
                float(actual["legacy"][i]),
                float((dg.min() + np.maximum(dl, 0).min()) / 2),
                places=6,
            )

    def test_normalization_and_threshold_ties(self):
        cal = np.array([1.0, 2.0, 3.0, 4.0])
        query = np.array([7.0])
        a, b, m = core.norm(cal, query, np.array([0, 1]))
        self.assertAlmostEqual(m["mean"], 1.5)
        self.assertAlmostEqual(float(b[0]), 11.0)
        threshold = np.quantile(cal, 0.95, method="higher")
        self.assertEqual(threshold, 4.0)
        self.assertFalse(bool((cal > threshold).any()))

    def test_evaluation_requires_complete_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                evaluate(Path(tmp), Path(tmp), [27042], "method")


if __name__ == "__main__":
    unittest.main()
