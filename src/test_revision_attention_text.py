"""Synthetic tests for authentic near-threshold captions and bounded prose."""

import unittest
from copy import deepcopy

import numpy as np

from revision_attention_text import make_appendix_fragment, scientific_latex, verify_cases


class AttentionTextTests(unittest.TestCase):
    def evidence(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([.5000001, .5001, .51, .4999999, .4999, .49])
        cases = {}
        for kind, start, label in (("FP", 0, 0), ("FN", 3, 1)):
            for rank in (1, 2, 3):
                index = start + rank - 1
                cases[f"{kind}_{rank}"] = {"sample_idx": index, "score": float(scores[index]),
                                           "distance_to_threshold": abs(float(scores[index]) - .5),
                                           "attention_forward_score": float(scores[index]), "true_label": label}
        return {"threshold": .5, "individual_cases": cases}, labels, scores

    def test_captions_preserve_nonzero_distance_after_six_decimal_score_rounding(self):
        summary, labels, scores = self.evidence()
        verified = verify_cases(summary, labels, scores, .5)
        text = make_appendix_fragment(verified, 31)
        self.assertIn(r"1.000\times10^{-7}", text)
        self.assertIn("p=0.500000", text)
        self.assertNotIn("within $0.001$", text)
        self.assertEqual(text.count(r"\begin{figure}[H]"), 6)
        self.assertIn("dotted vertical line", text)

    def test_wrong_original_label_is_rejected(self):
        summary, labels, scores = self.evidence()
        summary["individual_cases"]["FP_1"]["true_label"] = 1
        with self.assertRaisesRegex(ValueError, "Incorrect saved error class or label"):
            verify_cases(summary, labels, scores, .5)

    def test_reordered_cases_are_not_presented_as_nearest(self):
        summary, labels, scores = self.evidence()
        summary["individual_cases"]["FP_1"] = deepcopy(summary["individual_cases"]["FP_2"])
        with self.assertRaisesRegex(ValueError, "not the declared nearest"):
            verify_cases(summary, labels, scores, .5)

    def test_changed_score_is_rejected(self):
        summary, labels, scores = self.evidence()
        summary["individual_cases"]["FN_3"]["score"] += 1e-9
        with self.assertRaisesRegex(ValueError, "Displayed score differs"):
            verify_cases(summary, labels, scores, .5)

    def test_changed_attention_forward_error_class_is_rejected(self):
        summary, labels, scores = self.evidence()
        summary["individual_cases"]["FN_1"]["attention_forward_score"] = .5000001
        with self.assertRaisesRegex(ValueError, "changes the threshold error class"):
            verify_cases(summary, labels, scores, .5)

    def test_scientific_distance_does_not_accept_nonfinite_values(self):
        with self.assertRaises(ValueError):
            scientific_latex(float("nan"))
        self.assertEqual(scientific_latex(0), "0")


if __name__ == "__main__":
    unittest.main()
