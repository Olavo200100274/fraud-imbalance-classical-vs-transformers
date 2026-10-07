"""Synthetic checks for bounded, correctly selected attention diagnostics."""

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import torch

from attention_analysis import classify_predictions, select_hard_cases, summarise_cls_attention
import attention_analysis
from models.fttransformer import FTTransformer


class AttentionDiagnosticTests(unittest.TestCase):
    def test_hard_cases_are_nearest_actual_errors(self):
        labels, unused = classify_predictions(np.array([0, 0, 1, 1, 1, 0]),
                                               np.array([.7, .51, .49, .1, .9, .1]), .5)
        selected = select_hard_cases(labels, np.array([.7, .51, .49, .1, .9, .1]), .5, n=1)
        np.testing.assert_array_equal(selected["FP"], [1])
        np.testing.assert_array_equal(selected["FN"], [2])

    def test_uniform_entropy_includes_cls_token(self):
        summary = summarise_cls_attention(np.full((4, 31), 1 / 31), 25, 5)
        self.assertAlmostEqual(summary["normalized_entropy"], 1)
        self.assertAlmostEqual(summary["mean_categorical_attention"], 1 / 31)
        self.assertAlmostEqual(summary["ratio_cat_over_num"], 1)

    def test_numerical_only_schema_does_not_invent_categorical_values(self):
        summary = summarise_cls_attention(np.full((2, 4), .25), 3, 0)
        self.assertIsNone(summary["mean_categorical_attention"])
        self.assertIsNone(summary["ratio_cat_over_num"])

    def test_non_normalised_attention_is_rejected(self):
        with self.assertRaises(ValueError):
            summarise_cls_attention(np.full((2, 4), .2), 3, 0)

    def test_invalid_token_schema_is_rejected(self):
        with self.assertRaises(ValueError):
            summarise_cls_attention(np.full((2, 4), .25), 2, 0)

    def test_cli_coordinates_baf_ram_before_extraction(self):
        arguments = ["attention_analysis.py", "--dataset", "baf_base", "--run-dir", "pinned_run",
                     "--output-dir", "results_revision/test/attention", "--device", "cpu"]
        with patch("sys.argv", arguments), \
                patch("revision_resources.baf_training_resource", return_value=nullcontext()) as resource, \
                patch.object(attention_analysis, "run_analysis") as extract:
            attention_analysis.main()
        self.assertEqual(resource.call_args.args[0], "baf_base")
        extract.assert_called_once()

    def test_cli_rejects_preserved_outputs_before_lock_or_data_loading(self):
        arguments = ["attention_analysis.py", "--dataset", "baf_base", "--run-dir", "pinned_run",
                     "--output-dir", "results/unsafe_attention", "--device", "cpu"]
        with patch("sys.argv", arguments), \
                patch("revision_resources.baf_training_resource") as resource, \
                patch.object(attention_analysis, "run_analysis") as extract:
            with self.assertRaisesRegex(ValueError, "preserved"):
                attention_analysis.main()
        resource.assert_not_called()
        extract.assert_not_called()


class AttentionForwardRegressionTests(unittest.TestCase):
    """Small synthetic inference only; no saved scientific data or fitting."""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def check_forward_equivalence(self, device, *, categorical, blocks, rows=17,
                                  final_norm=False):
        torch.manual_seed(42)
        cards = [3, 5] if categorical else []
        model = FTTransformer(
            d_numerical=5, cat_cardinalities=cards, d_token=32,
            n_blocks=blocks, n_heads=4, d_ffn=48,
            attention_dropout=.3, ffn_dropout=.4, residual_dropout=.2,
        ).to(device).eval()
        if final_norm:
            model.transformer.norm = torch.nn.LayerNorm(32).to(device).eval()
        x_num = torch.randn(rows, 5, device=device)
        x_cat = None
        if categorical:
            x_cat = torch.stack([torch.arange(rows, device=device) % card
                                 for card in cards], dim=1)
            x_cat[0, 0] = -1
            x_cat[-1, 1] = cards[1] + 8
        before = {name: value.clone() for name, value in model.state_dict().items()}
        modes = [module.training for module in model.modules()]
        cpu_rng = torch.get_rng_state().clone()
        device_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        with torch.no_grad():
            expected = model(x_num, x_cat)
            actual, weights = model.forward_with_attention(x_num, x_cat)
            repeated, repeated_weights = model.forward_with_attention(x_num, x_cat)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
        self.assertEqual(len(weights), blocks)
        token_count = 1 + 5 + len(cards)
        for values, again in zip(weights, repeated_weights):
            self.assertEqual(tuple(values.shape), (rows, token_count, token_count))
            self.assertFalse(values.requires_grad)
            self.assertTrue(torch.isfinite(values).all().item())
            self.assertTrue((values >= 0).all().item())
            torch.testing.assert_close(values.sum(dim=-1),
                                       torch.ones(rows, token_count, device=device),
                                       rtol=0, atol=1e-6)
            torch.testing.assert_close(values, again, rtol=0, atol=0)
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertEqual(set(before), set(model.state_dict()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
        if device_rng is not None:
            self.assertTrue(torch.equal(torch.cuda.get_rng_state(device), device_rng))
        if categorical:
            reserved = x_cat.clone()
            for column, card in enumerate(cards):
                invalid = (reserved[:, column] < 0) | (reserved[:, column] >= card)
                reserved[invalid, column] = card
            reserved_logits, reserved_weights = model.forward_with_attention(x_num, reserved)
            torch.testing.assert_close(reserved_logits, actual, rtol=0, atol=0)
            for values, mapped in zip(weights, reserved_weights):
                torch.testing.assert_close(values, mapped, rtol=0, atol=0)

    def test_cpu_numerical_only_multiple_layers(self):
        self.check_forward_equivalence(torch.device("cpu"), categorical=False, blocks=2)

    def test_cpu_unknown_categories_multiple_layers(self):
        self.check_forward_equivalence(torch.device("cpu"), categorical=True, blocks=3)

    def test_cpu_single_case_with_encoder_final_normalisation(self):
        self.check_forward_equivalence(torch.device("cpu"), categorical=True, blocks=2,
                                       rows=1, final_norm=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_matches_normal_forward_and_preserves_state(self):
        self.check_forward_equivalence(torch.device("cuda"), categorical=True, blocks=3,
                                       rows=2048)

    def test_diagnostic_explicitly_disables_dropout(self):
        model = FTTransformer(2, [], 16, 2, 4, 24, .3, .4, .2).train()
        x_num = torch.tensor([[.1, -.7], [.3, .8]])
        before = {name: value.clone() for name, value in model.state_dict().items()}
        actual, weights = model.forward_with_attention(x_num)
        self.assertTrue(all(not module.training for module in model.modules()))
        with torch.no_grad():
            expected = model(x_num)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for values in weights:
            torch.testing.assert_close(values.sum(dim=-1), torch.ones(2, 3), rtol=0, atol=1e-6)

    def test_post_normalised_encoder_is_rejected_before_mode_change(self):
        model = FTTransformer(2, [], 16, 2, 4, 24, .3, .4, .2).train()
        model.transformer.layers[1].norm_first = False
        modes = [module.training for module in model.modules()]
        with self.assertRaisesRegex(ValueError, "Pre-LN"):
            model.forward_with_attention(torch.ones(2, 2))
        self.assertEqual(modes, [module.training for module in model.modules()])


if __name__ == "__main__":
    unittest.main()
