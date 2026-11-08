from __future__ import annotations

import unittest

from protein_sae_pipeline.scoring.identity import (
    calculate_positional_identity,
    normalize_sequence,
    positional_sequence_identity,
)


class PositionalIdentityTests(unittest.TestCase):
    def test_normalizes_case_and_whitespace(self) -> None:
        self.assertEqual(normalize_sequence(" acd\nEFG "), "ACDEFG")
        result = calculate_positional_identity("ACDEFG", " acd\nEFG ")
        self.assertEqual(result.aligned_length, 6)
        self.assertEqual(result.matches, 6)
        self.assertEqual(result.mismatches, 0)
        self.assertEqual(result.fraction, 1.0)
        self.assertEqual(result.percent, 100.0)

    def test_is_exact_positional_identity(self) -> None:
        result = calculate_positional_identity("AAAAAAAAAA", "AAACCCCCCC")
        self.assertEqual(result.matches, 3)
        self.assertEqual(result.mismatches, 7)
        self.assertEqual(result.fraction, 0.3)
        self.assertEqual(result.percent, 30.0)
        self.assertEqual(positional_sequence_identity("AAAA", "AACA"), 0.75)

    def test_rejects_invalid_inputs(self) -> None:
        cases = [
            ("", "", "empty"),
            ("AAAA", "AAA", "equal lengths"),
            ("AAAA", "AA-A", "noncanonical"),
            ("AAAA", "AAXA", "noncanonical"),
            ("AAAA", "AA*A", "noncanonical"),
            ("AAAA", "AAAA*", "noncanonical"),
        ]
        for reference, candidate, message in cases:
            with self.subTest(reference=reference, candidate=candidate):
                with self.assertRaisesRegex(ValueError, message):
                    calculate_positional_identity(reference, candidate)


if __name__ == "__main__":
    unittest.main()
