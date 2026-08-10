import unittest

import numpy as np

from quantum.non_classicality import (
    mixed_state_magic_witness_qubit_from_expectations,
    stabilizer_renyi_entropy_qubit_from_expectations,
)


class MixedStateMagicWitnessTests(unittest.TestCase):
    def test_pure_stabilizer_state_has_zero_witness(self):
        result = mixed_state_magic_witness_qubit_from_expectations(0.0, 0.0, 1.0)

        self.assertTrue(np.isclose(result["purity"], 1.0))
        self.assertTrue(np.isclose(result["M_alpha"], 0.0))
        self.assertTrue(np.isclose(result["W_alpha"], 0.0))
        self.assertFalse(bool(result["is_magic_exact"]))

    def test_pure_magic_state_is_detected(self):
        value = 1.0 / np.sqrt(3.0)
        result = mixed_state_magic_witness_qubit_from_expectations(
            value, value, value, alpha=2
        )

        self.assertTrue(np.isclose(result["purity"], 1.0))
        self.assertGreater(result["W_alpha"], 0.0)
        self.assertTrue(bool(result["is_magic_exact"]))

    def test_mixed_stabilizer_exposes_false_positive_of_old_M2(self):
        px, py, pz = 0.5, 0.0, 0.0
        old_M2, _, _, _, _ = stabilizer_renyi_entropy_qubit_from_expectations(
            px, py, pz
        )
        result = mixed_state_magic_witness_qubit_from_expectations(
            px, py, pz, alpha=2
        )

        self.assertGreater(old_M2, 0.0)
        self.assertLess(result["W_alpha"], 0.0)
        self.assertFalse(bool(result["is_magic_exact"]))

    def test_half_order_witness_is_exact_octahedron_test(self):
        px = np.array([0.5, 1.0 / np.sqrt(2.0)])
        py = np.array([0.0, 1.0 / np.sqrt(2.0)])
        pz = np.zeros(2)
        result = mixed_state_magic_witness_qubit_from_expectations(
            px, py, pz, alpha=0.5
        )

        expected = 2.0 * np.log((1.0 + np.abs(px) + np.abs(py)) / 2.0)
        self.assertTrue(np.allclose(result["W_alpha"], expected))
        self.assertTrue(
            np.array_equal(result["is_magic_exact"], np.array([False, True]))
        )

    def test_new_M2_reproduces_previous_formula(self):
        rng = np.random.default_rng(1234)
        bloch = rng.normal(size=(100, 3))
        bloch /= np.maximum(np.linalg.norm(bloch, axis=1, keepdims=True), 1.0)
        px, py, pz = bloch.T

        old_M2, _, _, _, _ = stabilizer_renyi_entropy_qubit_from_expectations(
            px, py, pz
        )
        result = mixed_state_magic_witness_qubit_from_expectations(
            px, py, pz, alpha=2
        )

        self.assertTrue(
            np.allclose(result["M_alpha"], old_M2, atol=1e-13, rtol=1e-13)
        )


if __name__ == "__main__":
    unittest.main()
